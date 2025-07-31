# Simple Emotion Classifier 😜
This project explores various techniques and models for building a robust text sentiment classifier. It covers the full workflow from data preprocessing to model training, evaluation, and real-time prediction.

## Overview
The primary goal of this project is to build an effective sentiment classifier by experimenting with a range of text representation techniques and machine learning/deep learning models.

This initial phase focuses on establishing a baseline with one of the most traditional and widely used methods:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization.
- Logistic Regression as the classification model.

## Getting Started

### 1. Prerequisites
- Python 3.11
- Git

### 2. Installation
- Clone the repo:
```bash
```git clone https://github.com/MinyoungSeol/simple-sentiment-classifier.git
cd simple-sentiment-classifier
```

- Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

- Install dependencies:
```bash
pip install -r requirements.txt
# or you can do it manually by typing:
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn nltk joblib
python -m nltk.downloader all
```

- Download NLTK data (These downloads will happen automatically when running the scripts, but you can also run them manually if needed): 
```Python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Data Preparation
Download the original dataset 'training.1600000.processed.noemoticon.csv' and place it into the data/ directory.

Link: https://www.kaggle.com/datasets/kazanova/sentiment140

## Running the Workflow

### 1. Preprocess raw data & sample:
```bash
python preprocess_data.py
```

### 2. Train, Evaluate, and Save model:
Run the .py file anything you want to under /notebooks dir. For now, this should be the tfidf-logistic-regression.py

### 3. Interactive Sentiment Test:
Run any of the .py file under /notebooks/test dir, for an interactive real-time prediction of your input!
Type 'quit' or 'exit' to stop.

## Current Progress & Future Enhancements

This project currently showcases the implementation of a TF-IDF + Logistic Regression based sentiment classifier. This serves as our initial baseline and a proof-of-concept for the broader sentiment analysis project.

In the next phases, we plan to explore and integrate more advanced techniques and models to further enhance classification performance and generalize the classifier's capabilities. This includes:

- Exploring other text embedding techniques (e.g., Word2Vec, GloVe, FastText).

- Implementing deep learning models (e.g., LSTM, GRU, Transformers like BERT).

- Performing hyperparameter tuning for optimal model performance.

- Evaluating various ensemble methods.