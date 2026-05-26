# Text-Classification with Tensorflow

IMDB Sentiment Analysis
A simple yet effective deep learning model for binary sentiment classification (positive/negative) of movie reviews using the IMDB dataset. The notebook demonstrates two approaches:
1. **Custom Embedding + Dense Network** – Trains a word embedding from scratch.
2. **TensorFlow Hub Pre‑trained Embedding** (NNLM) – A lightweight demonstration of transfer learning.

The final model achieves **≈85% validation accuracy** after only 5 epochs.
---

##  Overview :
The IMDB dataset contains 25,000 highly polarised movie reviews for training and 25,000 for testing. Each review is a sequence of word indices (integers). The notebook:
1. Loads the IMDB dataset (top 20,000 words, max length 200).
2. Converts integer sequences to fixed‑length vectors using **padding**.
3. Builds a sequential model:
   - `Embedding` layer (20,000 vocabulary × 128 embedding dim)
   - `GlobalAveragePooling1D` (aggregates word vectors)
   - `Dense` 128 (ReLU)
   - `Dropout` 0.5 (regularisation)
   - `Dense` 1 (sigmoid output)
4. Compiles with `binary_crossentropy` loss and `adam` optimizer.
5. Trains for 5 epochs (batch size 32) and evaluates on the test set.
6. Also shows how to use a TF‑Hub pre‑trained text embedding (`nnlm-en-dim128`) for zero‑shot feature extraction.

---
##  How to Run :

### 1. Install dependencies

```bash
pip install tensorflow tensorflow-hub
```

### 2. Open the notebook

```bash
jupyter notebook "Text Classification.ipynb"
```

Or run in **Google Colab** – the notebook is fully self‑contained.

### 3. Execute cells sequentially

The notebook will automatically download the IMDB dataset via `tensorflow.keras.datasets`.

---

##  Possible Improvements :

- **Increase epochs** – more training (e.g., 10–15) with early stopping could push accuracy to 88‑90%.
- **Use pre‑trained embeddings** – GloVe, FastText, or the TF‑Hub NNLM layer (shown in the first cells) may improve generalisation.
- **Add bidirectional LSTM** – replaces average pooling for better context modelling.
- **Hyperparameter tuning** – dropout rate, embedding dimension, number of dense units.
- **Data augmentation** – random shuffling of words (not typical for sentiment) or synonym replacement.

---

##  Repository Structure :

```
.
├── Text Classification.ipynb   # Main notebook
└── README.md                   # This file
```
