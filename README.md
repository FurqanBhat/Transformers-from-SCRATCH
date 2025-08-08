# Transformer Model from Scratch for English-to-Italian Translation

This project is a complete implementation of the Transformer architecture from the seminal paper "Attention Is All You Need," built from scratch using PyTorch. The model is designed and trained for machine translation, specifically for translating text from **English to Italian**.

The entire model, including all its core components like Multi-Head Attention, Positional Encoding, and the Encoder-Decoder stacks, is built from basic PyTorch modules, without relying on pre-built `nn.Transformer` layers.

---

## 🚀 Features

* **From-Scratch Implementation**: The Transformer model is built from the ground up, providing a clear and educational look into its internal workings.
* **Machine Translation**: Trained on the Opus Books dataset for English-to-Italian translation.
* **Custom Tokenizer**: A custom `WordLevel` tokenizer is built from the training data using the Hugging Face `tokenizers` library.
* **Comprehensive Training Loop**: Includes a full training pipeline with a validation loop, loss calculation, and Adam optimizer.
* **Performance Logging**: Integrated with TensorBoard to log training loss and validation metrics like Character Error Rate (CER), Word Error Rate (WER), and BLEU score.
* **Inference with Greedy Decode**: A `greedy_decode` function is implemented to perform inference and generate translations from the trained model.

---

## ⚙️ Model Architecture

The model adheres to the original Transformer architecture, consisting of:
* **Input Embeddings**: Converts input tokens into vectors of dimension `d_model`.
* **Positional Encoding**: Injects positional information into the embeddings using sine and cosine functions.
* **Encoder Stack**: Composed of 6 identical Encoder blocks. Each block contains:
    * A **Multi-Head Self-Attention** layer.
    * A **Feed-Forward Neural Network**.
    * Residual connections and layer normalization are applied around each of the two sub-layers.
* **Decoder Stack**: Composed of 6 identical Decoder blocks. Each block contains:
    * A **Masked Multi-Head Self-Attention** layer to prevent attending to future tokens.
    * A **Multi-Head Cross-Attention** layer that attends to the output of the encoder stack.
    * A **Feed-Forward Neural Network**.
    * Residual connections and layer normalization are applied around each of the three sub-layers.
* **Projection Layer**: A final linear layer followed by a log-softmax to produce the output probabilities over the target vocabulary.

---

## 📚 Dataset

The model is trained on the **Opus Books** dataset, an English-Italian parallel corpus sourced from the Hugging Face Hub (`opus_books`, `en-it` configuration).

The dataset is split into a 90% training set and a 10% validation set. Custom `WordLevel` tokenizers for both English and Italian are built and trained on this dataset.

* **Max Source Sentence Length**: 309 tokens
* **Max Target Sentence Length**: 274 tokens

---

## 🏋️‍♀️ Training

The model was trained for **20 epochs** on a CUDA-enabled GPU. The key hyperparameters used for training are detailed below:

| Hyperparameter      | Value      |
| :------------------ | :--------- |
| `batch_size`        | 8          |
| `num_epochs`        | 20         |
| `learning_rate`     | `1e-4`     |
| `seq_len`           | 350        |
| `d_model`           | 512        |
| `d_ff`              | 2048       |
| `num_heads` (h)     | 8          |
| `num_blocks` (N)    | 6          |
| `dropout`           | 0.1        |

* **Optimizer**: Adam with $\epsilon = 10^{-9}$
* **Loss Function**: Cross-Entropy Loss with label smoothing of 0.1. The padding token is ignored during loss calculation.

---

## 📊 Training Output & Results

The model's performance improved steadily over the 20 epochs. Below are selected examples from the validation run at the end of each epoch, showing the source text, the ground truth target, and the model's prediction.

### Epoch 00

* **SOURCE**: "The child ought to have change of air and scene," he added, speaking to himself; "nerves not in a good state."
* **TARGET**: — Per questa bimba ci vorrebbe un cambiamento d'aria e di luogo — aggiunse come se parlasse a sè stesso. I suoi nervi non sono in buono stato.
* **PREDICTED**: — , e la mia voce , — disse , — disse la mia vita , — non è più .

### Epoch 04

* **SOURCE**: “No shoot,” says Friday, “no yet; me shoot now, me no kill; me stay, give you one more laugh:” and, indeed, so he did; for when the bear saw his enemy gone, he came back from the bough, where he stood, but did it very cautiously, looking behind him every step, and coming backward till he got into the body of the tree, then, with the same hinder end foremost, he came down the tree, grasping it with his claws, and moving one foot at a time, very leisurely.
* **TARGET**: Venerdì saltò tanto, e i corrispondenti atti dell’orso furono tanto grotteschi, che avemmo campo a ridere per un bel pezzo Dopo di che andato all’estrema punta del ramo, laddove poteva farlo piegare col proprio peso vi si attaccò, e lasciandosi bellamente calar giù finchè fosse vicino a terra abbastanza per ispiccare un salto, eccolo su due piedi e presso al suo moschetto di cui si munì, ma lasciandolo tuttavia ozioso. — «Orsù dunque, Venerdì, gli diss’io, che cosa state a fare adesso? Perchè non gli tirate? — Non ancora; adesso allora, ripetè; me non ammazzare lui adesso, me fermarmi qui; me darvi sempre più bel ridere».
* **PREDICTED**: « Oh , Venerdì , che non mi , nè mi , e mi più di nuovo , e che a me , quando gli occhi di nuovo , quando , quando , quando , quando , quando , quando , quando , quando , , , si , e , , e , , , e , , , , , e , , , , e , , , e , e , , e , e , , , , , , e , e , , , , , , , , , e , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,

### Epoch 09

* **SOURCE**: Already in February he had received a letter from Mary Nikolavna to say that his brother Nicholas's health was getting worse, but that he would not submit to any treatment. In consequence of this news Levin went to Moscow, saw his brother, and managed to persuade him to consult a doctor and go to a watering-place abroad.
* **TARGET**: Inoltre, in febbraio, aveva ricevuta da Mar’ja Nikolaevna una lettera in cui si diceva che le condizioni di salute del fratello erano peggiorate, e che egli non voleva curarsi; in seguito a questa lettera, Levin era andato a Mosca e aveva fatto in tempo a persuadere il fratello a consigliarsi con un medico e ad andare all’estero per la cura delle acque.
* **PREDICTED**: Durante il tè , era rimasto da una lettera da Mar ’ ja Nikolaevna che aveva parlato del fratello Nikolaj , ma che non temeva che fosse la notizia di Levin non potesse più la notizia di Mosca , e Levin si era allontanato verso di lui , e per un viaggio di cento rubli di un viaggio .

### Epoch 14

* **SOURCE**: This beautiful baby only inspired him with a sense of repulsion and pity.
* **TARGET**: Quel bellissimo bambino gli ispirava soltanto disgusto e pena.
* **PREDICTED**: Questo bambino solo con lui la propria impressione e con la pena .

### Epoch 19

* **SOURCE**: George got hold of the paper, and read us out the boating fatalities, and the weather forecast, which latter prophesied "rain, cold, wet to fine" (whatever more than usually ghastly thing in weather that may be), "occasional local thunder-storms, east wind, with general depression over the Midland Counties (London and Channel).
* **TARGET**: Giorgio s’impadronì del giornale, e ci lesse le disgrazie fluviali e marittime, e la previsione del tempo, che vaticinava «pioggia, freddo, vento» (tutto ciò che ci può esser di peggiore nel tempo), e qualche temporale locale, con depressione generale sulle contee centrali (Londra e Canale).
* **PREDICTED**: Giorgio della carta , e ci dirigemmo a sentire la barca , e i racconti delle barche che nelle pozzanghere , nelle notti più lunghe ( più si può dire che si può fare una qualche tempo ), che si può scegliere una e un po ’ di sole , più durante la stagione delle torte , un po ’ più di sole .

---

## 📜 How to Use

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

**2. Install dependencies:**
It is recommended to use a virtual environment. Create a `requirements.txt` file with the content below and run:
```bash
pip install -r requirements.txt
```

**3. Run the Jupyter Notebook:**
The notebook `transformers-from-scratch.ipynb` contains all the code needed to download the dataset, build the tokenizers, define the model, and run the training process. You can execute the cells sequentially in a Jupyter environment like Jupyter Lab or VS Code.

**4. (Optional) Use Pre-trained Weights:**
The trained model weights (`.pt` files) are saved in the `weights/` directory after each epoch. You can load a specific model checkpoint by setting the `preload` value in the configuration dictionary at the top of the notebook.

---

## 📦 Dependencies

The project relies on the following Python libraries:

* `torch`
* `torchmetrics`
* `transformers`
* `datasets`
* `tokenizers`
* `tqdm`


