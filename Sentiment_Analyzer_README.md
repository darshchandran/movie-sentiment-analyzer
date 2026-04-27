# Movie Sentiment Analyzer

I built this proper NLP project to understand how machines can interpret human language and emotion. The idea is simple — given a movie review, can a model figure out if the person liked the movie or not?

---

## What it does

Takes any movie review as plain text input and classifies it as **positive** or **negative**, along with a confidence score.

```
Input:  "The film was a masterpiece. Incredible performances throughout."
Output: Positive (96.3% confidence)

Input:  "Waste of time. Plot made no sense and the ending was terrible."
Output: Negative (91.7% confidence)
```

---

## Why I built it

Wanted to get hands-on with the full ML pipeline — data cleaning, feature extraction, model training, evaluation. Sentiment analysis felt like the right starting point because the problem is intuitive (you already know what positive and negative means) but the solution involves real ML concepts that carry across every NLP project.

---

## Dataset

**IMDb Movie Reviews Dataset** — 50,000 reviews, evenly split between positive and negative.  
Available on Kaggle: [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Tech Stack

- **Python 3.11**
- **Pandas** — loading and manipulating the dataset
- **NLTK** — text preprocessing (stopword removal, tokenization)
- **Scikit-learn** — TF-IDF vectorization, Logistic Regression, evaluation metrics
- **Matplotlib / Seaborn** — visualizations
- **Jupyter Notebook** — development environment

---

## How it works

### 1. Load the data
50,000 reviews pulled in with Pandas. Each row has a review (text) and a label (positive/negative).

### 2. Clean the text
Raw reviews are messy — HTML tags, punctuation, numbers, filler words like "the", "is", "a". Stripped all of that out using NLTK so the model only sees words that actually carry meaning.

```python
# Example of what cleaning does
Before: "<br />This movie was SO good!! Loved every minute of it :)"
After:  "movie good loved every minute"
```

### 3. TF-IDF Vectorization
Computers can't read words — only numbers. TF-IDF converts each review into a vector where words that are frequent in one review but rare across all reviews score higher (they're more meaningful).

### 4. Train the model
Fed the vectors + labels into a Logistic Regression classifier. It learns which word patterns lean positive and which lean negative.

### 5. Evaluate
Tested on 10,000 unseen reviews. Checked accuracy, plotted a confusion matrix, and ran a full classification report.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 88.4% |
| Precision | 88.6% |
| Recall | 88.2% |
| F1 Score | 88.4% |

The model struggles a bit with sarcasm ("oh wow, what a *great* film") — which is honestly a hard problem even for humans.

---

## How to run it

```bash
# Clone the repo
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn nltk jupyter

# Launch the notebook
jupyter notebook sentiment_analyzer.ipynb
```

Download the dataset from the Kaggle link above and place `IMDB Dataset.csv` in the root folder before running.

---

## Project structure

```
sentiment-analyzer/
│
├── sentiment_analyzer.ipynb    # Main notebook — run this
├── IMDB Dataset.csv            # Dataset (download from Kaggle)
├── requirements.txt
└── README.md
```

---

## What I learned

- Full NLP preprocessing pipeline from scratch
- How TF-IDF works and why it beats simple word counts
- Logistic Regression for binary classification
- Reading and interpreting a confusion matrix
- Why clean data matters more than model choice

---

## What's next

- Try BERT or a transformer-based model for better sarcasm handling
- Build a simple web interface using Streamlit so anyone can type a review and get a prediction
- Extend to multi-class (very positive / positive / neutral / negative / very negative)
