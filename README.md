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

**Note**: Advanced models require significant computational resources. GPU is highly recommended.

- **CPU training**: Will work but slow (several hours)
- **GPU training**: Much faster (30-60 minutes for all models)

The notebook will automatically detect and use CUDA if available.

## Model Performance

### Baseline Models (main.ipynb)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~79% | ~0.76 |
| Naive Bayes | ~78% | ~0.74 |
| Random Forest | ~77% | ~0.73 |

### Advanced Models (advanced_models.ipynb)

| Model | Parameters | F1 Score |
|-------|-----------|----------|
| LSTM | ~3M | ~0.80 |
| GRU | ~3M | ~0.80 |
| DistilBERT | ~66M | ~0.83 |
| RoBERTa | ~125M | ~0.84 |
| **Ensemble** | - | **~0.85** |

*Note: Exact scores may vary depending on random initialization and hardware.*

## Training Tips

### For CPU Users
- Reduce batch size (e.g., 8 or 16)
- Reduce number of epochs
- Use smaller models (LSTM/GRU first)
- Consider using DistilBERT instead of RoBERTa

### For GPU Users
- Can use larger batch sizes (32 or 64)
- Train all models for full comparison
- Consider training multiple runs with different seeds

### Memory Optimization
If you encounter OOM errors:
- Reduce batch size
- Reduce max sequence length
- Use gradient accumulation
- Train models separately instead of in one session

## Output Files

The notebooks will generate:

- `submission.csv` - Baseline model predictions (from main.ipynb)
- `submission_ensemble.csv` - Ensemble predictions (from advanced_models.ipynb)
- `best_lstm_model.pt` - Best LSTM checkpoint
- `best_gru_model.pt` - Best GRU checkpoint
- `best_distilbert_model.pt` - Best DistilBERT checkpoint
- `best_roberta_model.pt` - Best RoBERTa checkpoint

## Next Steps for Improvement

1. **Cross-Validation**: Implement k-fold CV for more robust evaluation
2. **Hyperparameter Tuning**: Use Optuna or Ray Tune for automated HP search
3. **External Data**: Incorporate additional disaster-related datasets
4. **Semi-Supervised Learning**: Use unlabeled data for pre-training
5. **Advanced Ensembling**: Try stacking or blending techniques
6. **Custom Architectures**: Experiment with attention mechanisms
7. **Post-Processing**: Implement confidence-based thresholding

## Troubleshooting

### Common Issues

**Issue**: Kaggle API 401 Unauthorized
- Solution: Regenerate your Kaggle API token and ensure permissions are set correctly

**Issue**: CUDA out of memory
- Solution: Reduce batch size or use gradient accumulation

**Issue**: Transformers library errors
- Solution: Update transformers: `pip install --upgrade transformers`

**Issue**: NLTK data not found
- Solution: Run `nltk.download('all')` or download specific packages

## Resources

- [Competition Page](https://www.kaggle.com/competitions/nlp-getting-started)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [NLTK Documentation](https://www.nltk.org/)

## License

This project is for educational purposes as part of the Kaggle competition.

## Acknowledgments

- Kaggle for hosting the competition
- Hugging Face for the Transformers library
- PyTorch team for the deep learning framework
- NLTK team for NLP tools
