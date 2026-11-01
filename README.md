# Sentiment-Analysis# Sentiment Analysis Using PyTorch

A Deep Learning-based Sentiment Analysis project that classifies movie reviews as **Positive** or **Negative** using PyTorch and the IMDb movie review dataset.

## Project Overview

This project implements a custom Neural Network using PyTorch to perform binary sentiment classification on movie reviews.

The model processes raw text, converts words into numerical representations using a custom vocabulary, generates embeddings, and predicts the sentiment of a review.

**Project Type:** Natural Language Processing (NLP)  
**Task:** Binary Sentiment Classification  
**Framework:** PyTorch  
**Dataset:** IMDb Movie Reviews  
**Execution Device:** CPU

## Objectives

- Build a sentiment classification model from scratch using PyTorch.
- Understand text preprocessing and tokenization.
- Implement word embeddings and mean pooling.
- Train a neural network using IMDb reviews.
- Save and load trained model weights.
- Perform sentiment predictions on new movie reviews.

## Dataset

**IMDb Movie Reviews Dataset**

Source: [Stanford NLP IMDb Dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)

The dataset contains movie reviews labeled as:

- 0 – Negative
- 1 – Positive

The dataset is loaded using Hugging Face Datasets.

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

## Model Architecture

The project uses a custom neural network built with PyTorch.

### Architecture

Input Text → Tokenization → Word Embedding → Mean Pooling → Dropout → Linear Layer → Sigmoid Probability → Sentiment Prediction

### Model Components

| Component | Configuration |
|---|---|
| Embedding Layer | 64 dimensions |
| Padding Index | 0 |
| Pooling | Masked Mean Pooling |
| Dropout | 0.3 |
| Output Layer | Linear (64 → 1) |
| Loss Function | BCEWithLogitsLoss |
| Optimizer | Adam |

### Training Configuration

| Parameter | Value |
|---|---|
| Maximum Vocabulary | 20,000 |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 64 |
| Batch Size | 32 |
| Learning Rate | 0.01 |
| Epochs | 50 |
| Optimizer | Adam |
| Device | CPU |
| Prediction Threshold | 0.5 |

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Hugging Face Datasets
- Jupyter Notebook

## Project Structure

```text
Sentiment-Analysis/
│
├── app.py
├── Requirements.txt
│
├── Sentiment_analysis_train.ipynb
├── Sentiment_analysis_test.ipynb
│
├── sentiment_model.pth
├── vocabulary.json
├── config.json
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nitinkumardatascienceengineer-create/Sentiment-Analysis.git
```

Navigate into the project:

```bash
cd Sentiment-Analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r Requirements.txt
```

## Model Training

Open the training notebook:

`Sentiment_analysis_train.ipynb`

The notebook covers:

1. Dataset loading.
2. Text preprocessing.
3. Tokenization.
4. Vocabulary creation.
5. Sequence padding.
6. Dataset and DataLoader creation.
7. Neural network implementation.
8. Model training.
9. Validation loss monitoring.
10. Saving trained model weights.

## Saved Model Files

The project contains three important model artifacts:

| File | Purpose |
|---|---|
| sentiment_model.pth | Trained PyTorch model weights |
| vocabulary.json | Word-to-index vocabulary mapping |
| config.json | Model configuration and preprocessing settings |

These files are required to reconstruct the trained model and process new reviews consistently.

## Model Testing

Open:

`Sentiment_analysis_test.ipynb`

The testing notebook demonstrates loading the saved model, preprocessing new reviews and generating sentiment predictions.

## Prediction Example

Example input:

```python
reviews = [
    "This movie was absolutely fantastic!",
    "The movie was boring and disappointing."
]
```

Expected prediction format:

```text
Review 1: Positive
Review 2: Negative
```

The model also produces a probability score for its prediction.

## Future Improvements

- Improve model generalization through proper validation.
- Experiment with LSTM and BiLSTM architectures.
- Implement pretrained Word2Vec or GloVe embeddings.
- Compare performance with Transformer-based models.
- Improve model evaluation using Precision, Recall, F1-score and ROC-AUC.
- Deploy the model as a publicly accessible web application.

## Author

**Nitin Kumar**

Aspiring Data Scientist | Python | Machine Learning | Deep Learning | NLP

GitHub: [nitinkumardatascienceengineer-create](https://github.com/nitinkumardatascienceengineer-create)

---

If you find this project useful, consider giving the repository a star.
