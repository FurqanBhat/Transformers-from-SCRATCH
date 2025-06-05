import torch
import torch.nn as nn
from model import build_transformer
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path
from tqdm import tqdm
import warnings
import torchmetrics



def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx=tokenizer_src.token_to_id('[SOS]')
    eos_idx=tokenizer_src.token_to_id('[EOS]')

    #precompute the  encoder output and reuse it for every token we get from decoder
    encoder_output=model.encode(src, src_mask)
    #initialize the decoder input with the  sos token
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(src).to(device) #(batch,sos_token)

    while True:
        if decoder_input.size(1) == max_len:
            break

        #build mask for decoder input
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        #calculate output of decoder
        output=model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        #get next token
        prob=model.project(output[:,-1]) #we give only last token to projection layer

        #select the token with max. probability
        _, next_word=torch.max(prob, dim=1)
        decoder_input.cat([decoder_input, torch.empty(1,1).fill_(next_word.item()).to(device)], dim=1)
        if next_word==eos_idx:
            break

    return decoder_input.squeeze(0) #remove batch dim




def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count=0
    source_texts=[]
    expected=[]
    predicted=[]

    #size of the control window
    console_width=80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0)==1, "batch size must be 1 for validation"
            
            model_out=greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text=batch['src_text']
            target_text=batch['tgt_text']
            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            #we use print.msg instead of print so that it doesn not interfare with progress bar of tqdm
            print_msg('-'*console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {model_out_text}')

            if count==num_examples:
                break


    #if tensorboard enabled, we can send above metrics to tenserboard as well
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()




def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
        
def get_ds(config):
    ds_raw=load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    #build tokenizers
    tokenizer_src=get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #keep 90% data for training and 10% for validation
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size

    train_ds_raw, val_ds_raw=random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds=BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config['seq_len'])

    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src=max(max_len_src, len(src_ids))
        max_len_tgt=max(max_len_tgt, len(tgt_ids))

    print(f"max length for source sentence: {max_len_src}")
    print(f"max length for target sentence: {max_len_tgt}")

    
    train_dataloader=DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader=DataLoader(val_ds, batch_size=1, shuffle=True) #we want to process sentences one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model=build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device using {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt=get_ds(config)
    model=get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    #Tensorboard
    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(), config["lr"], eps=1e-9)

    initial_epoch=0
    global_step=0
    if config['preload']:
        model_filename=get_weights_file_path(config, config["preload"])
        print(f"preloading model {model_filename}")
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        
        batch_iterator=tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input=batch['encoder_input'].to(device)#(batch, seq_len)
            decoder_input=batch['decoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)#(batch,1,1,seq_len)
            decoder_mask=batch['decoder_mask'].to(device)#(batch,1,seq_len,seq_len)

            encoder_output=model.encode(encoder_input, encoder_mask)#(batch, seq_len, d_model)
            decoder_output=model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)#(batch, seq_len, d_model)
            proj_output=model.project(decoder_output)#(batch, seq_len, tgt_vocab_size)
            
            label=batch['label'].to(device) #(batch, seq_len)

            #(batch, seq_len, tgt_vocab_size)->(batch*seq_len, tgt_vocab_size)
            loss=loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(f'loss: {loss.item():6.3f}')

            #log the loss on tensorboard
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            #backpropagate the loss
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg:batch_iterator.write(msg), global_step, writer)
            global_step+=1

            #save model after every epoch
            model_filename=get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "global_step":global_step

            }, model_filename)



if __name__ =='__main__':
    # warnings.filterwarnings('ignore')
    config=get_config()
    train_model(config)


