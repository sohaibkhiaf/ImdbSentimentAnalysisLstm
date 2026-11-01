# IMDb Sentiment Analysis using LSTM (PyTorch & Flask)

![RNN-LSTM Architecture](RNN-LSTM.webp)

## Overview & Project Goal

The goal of this project is to build, train, and deploy a Deep Learning Natural Language Processing (NLP) model capable of classifying movie reviews from the IMDb dataset as either **Positive** or **Negative**.

Using a **Long Short-Term Memory (LSTM)** recurrent neural network implemented in **PyTorch**, the model captures sequential dependencies in text to understand sentiment. Additionally, the repository includes an interactive **Flask web application** that allows users to input custom movie reviews and receive real-time sentiment predictions along with confidence scores.

---

## Technologies & Tools Used

- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `DataLoader`, `TensorDataset`)
- **Natural Language Processing (NLP):** NLTK (`stopwords`), Regular Expressions (`re`), Custom Vocabulary & Tokenization
- **Data Manipulation & Analysis:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Web Deployment:** Flask (HTML templates, `pickle` for vocabulary serialization)
- **Hardware Acceleration:** CUDA support (automatically falls back to CPU if GPU is unavailable)

---

## Project Structure

```text
ImdbSentimentAnalysisLstm/
├── imdb_sentiment_analysis_app/       # Flask web application for inference
│   ├── checkpoints/
│   │   └── model_state_dict.pt        # Saved PyTorch model weights
│   ├── templates/
│   │   └── index.html                 # Web interface template
│   ├── app.py                         # Flask server & prediction route
│   ├── model.py                       # SentimentLSTM model definition
│   └── vocabulary.pkl                 # Serialized word-to-index dictionary
├── imdb_train_dataset.csv             # Training dataset
├── imdb_test_dataset.csv              # Testing dataset
├── main.py                            # End-to-end data preprocessing, training & evaluation script
├── RNN-LSTM.webp                      # Architecture diagram
└── README.md                          # Project documentation
```

---

## How the Model is Trained

The training pipeline in `main.py` follows a complete end-to-end machine learning workflow:

### 1. Data Preprocessing & Vocabulary Building
- **Text Cleaning:** Punctuation, special characters, and digits are stripped from each review using regular expressions (`preprocess_string`).
- **Stop-word Removal:** Common English stop words (e.g., *the, is, a, and*) are filtered out using NLTK's corpus.
- **Vocabulary Construction:** Word frequencies are counted across the dataset using `Counter`, and the top `MAX_VOCAB_SIZE = 120` most frequent words are mapped to integer IDs (`word_to_id`), reserving index `0` for padding.
- **Tokenization & Encoding:** Reviews are converted into sequences of integer token IDs, while labels are encoded as `1` (`positive`) and `0` (`negative`).
- **Outlier Removal & Left Padding:** Zero-length encoded reviews are filtered out. All remaining sequences are truncated or left-padded with zeros to a fixed length of `MAX_SEQ_LEN = 12`.

### 2. Dataset Splitting & Batching
- The preprocessed training data is split into **80% training** and **20% validation** sets.
- Wrapped into PyTorch `TensorDataset` and loaded via `DataLoader` with `BATCH_SIZE = 50` (shuffled during training).

### 3. Model Architecture (`SentimentLSTM`)
The neural network consists of the following layers:
1. **Embedding Layer (`nn.Embedding`):** Converts integer token indices (`vocab_size = 121`, `padding_idx = 0`) into dense vectors of size `embedding_dim = 128`.
2. **Stacked LSTM Layer (`nn.LSTM`):** A 2-layer LSTM (`num_layers = 2`, `hidden_dim = 128`, `dropout = 0.3`) that processes the embedded token sequences and extracts temporal features from the final time step (`lstm_output[:, -1, :]`).
3. **Dropout Layer (`nn.Dropout`):** Applies a `0.3` dropout probability for regularization to prevent overfitting.
4. **Fully Connected Layer (`nn.Linear`):** Maps the `128`-dimensional hidden state to a single raw logit output (`output_size = 1`), passed through a Sigmoid activation during inference.

### 4. Training & Optimization
- **Loss Function:** Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`), which combines a Sigmoid layer and BCELoss in a numerically stable manner.
- **Optimizer:** Adam optimizer (`torch.optim.Adam`) with a learning rate of `0.001`.
- **Gradient Clipping:** Uses `nn.utils.clip_grad_norm_` (`clip = 5`) during backpropagation to prevent the exploding gradient problem common in RNNs/LSTMs.
- **Epochs & Checkpointing:** Trained for `5` epochs with training/validation loss and accuracy tracked and plotted at each epoch. The trained weights are saved to `checkpoints/model_state_dict.pt`.

---

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install torch numpy pandas seaborn matplotlib nltk flask
```

Ensure the NLTK English stopwords corpus is downloaded:

```python
import nltk
nltk.download('stopwords')
```

### 1. Train & Evaluate the Model

Run the main training script from the root directory:

```bash
python main.py
```

This script will:
- Load and visualize the dataset distribution.
- Preprocess text, train the `SentimentLSTM` model, and plot Loss/Accuracy curves.
- Evaluate accuracy on `imdb_test_dataset.csv`.
- Open an interactive CLI prompt to test custom reviews (type `exit` to quit and save the model checkpoint).

### 2. Run the Flask Web Application

Navigate to the web app directory and start the Flask server:

```bash
cd imdb_sentiment_analysis_app
python app.py
```

Open your browser at `http://127.0.0.1:5000/` to enter movie reviews and see the predicted sentiment (**Positive** / **Negative**) and confidence score.

