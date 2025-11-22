# Disaster Tweets NLP Analysis

This project implements both baseline and advanced deep learning models for the Kaggle "Natural Language Processing with Disaster Tweets" competition.

## Competition Overview

The goal is to predict which tweets are about real disasters and which ones are not. This is a binary classification task.

**Competition Link**: https://www.kaggle.com/competitions/nlp-getting-started

## Project Structure

```
TweetNLPAnalysis/
├── main.ipynb                  # Baseline analysis with classical ML models
├── advanced_models.ipynb       # Advanced PyTorch & Transformer models
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── train.csv                  # Training data (download from Kaggle)
├── test.csv                   # Test data (download from Kaggle)
└── sample_submission.csv      # Submission format (download from Kaggle)
```

## Notebooks

### 1. main.ipynb - Baseline Analysis

This notebook includes:
- Comprehensive exploratory data analysis (EDA)
- Text preprocessing and cleaning
- Word frequency analysis and word clouds
- Feature engineering
- Classical ML models (Logistic Regression, Naive Bayes, Random Forest)
- TF-IDF vectorization
- Model comparison

**Best for**: Understanding the dataset and establishing baseline performance.

### 2. advanced_models.ipynb - Deep Learning Models

This notebook implements advanced techniques:

#### Advanced Preprocessing
- Contraction expansion
- Lemmatization with WordNet
- Emoji to text conversion
- Keyword and location feature integration

#### Data Augmentation
- Synonym replacement using WordNet
- Random word deletion
- Random word swapping

#### Models Implemented
1. **LSTM** - Bidirectional LSTM with attention mechanism
2. **GRU** - Bidirectional GRU
3. **DistilBERT** - Lightweight transformer model
4. **RoBERTa** - Robust transformer model

#### Ensemble Methods
- Simple averaging of model predictions
- Weighted voting based on validation F1 scores

#### Training Features
- Class weighting for imbalanced data
- Learning rate scheduling
- Gradient clipping
- Best model checkpointing
- Comprehensive metrics tracking

**Best for**: Achieving high performance with state-of-the-art models.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy matplotlib seaborn wordcloud
pip install nltk contractions
pip install torch torchvision torchaudio
pip install transformers tokenizers
pip install scikit-learn tqdm jupyter kaggle
```

### 2. Download NLTK Data

The notebooks will automatically download required NLTK data, but you can also do it manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 3. Get the Dataset

#### Option A: Using Kaggle API (Recommended)

1. Accept the competition rules at https://www.kaggle.com/competitions/nlp-getting-started
2. Set up your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Create a new API token
   - Place `kaggle.json` in `~/.kaggle/`
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

3. Download the data:
```bash
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip
```

#### Option B: Manual Download

1. Visit https://www.kaggle.com/competitions/nlp-getting-started/data
2. Download `train.csv`, `test.csv`, and `sample_submission.csv`
3. Place them in the project directory

## Usage

### Running Baseline Analysis

```bash
jupyter notebook main.ipynb
```

Run all cells to:
1. Load and explore the data
2. Perform EDA
3. Train classical ML models
4. Generate baseline predictions

### Running Advanced Models

```bash
jupyter notebook advanced_models.ipynb
```


- **CPU training**: Will work but slow (several hours)
- **GPU training**: Much faster (30-60 minutes for all models)
