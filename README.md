## Fake News Detection with Machine Learning
## Overview
This project aims to detect fake news articles using machine learning techniques. It leverages natural language processing (NLP) and classification algorithms to distinguish between genuine and fake news.

## Dataset

The project utilizes a kaggle dataset. The dataset can be found here: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets. The dataset consists of two CSV files -- Fake.csv and True.csv. Each article contains the following information: article title, text, subject and the date the article was published on. The data collected were cleaned and processed, however, the punctuations and mistakes that existed in the fake news were kept in the text.
The data preprocessing steps involve, lowercasing, tokenization, removing of stop words, and stemming or lemmatizing the text.

## Project Structure
data/: Contains the dataset files (True.csv, Fake.csv).
main.ipynb: Jupyter notebook containing the entire code for preprocessing, model training, and evaluation.

## Model Development
- Utilizes a logistic regression model for binary classification.
- Implements natural language processing (NLP) techniques for text preprocessing and feature extraction.
- Evaluates model performance using standard metrics such as accuracy, precision, recall, and F1-score

## Dependencies
- NumPy (numpy): For numerical operations and array manipulation.
- Pandas (pandas): For data manipulation and analysis, particularly with DataFrame structures.
- Matplotlib (matplotlib.pyplot): For creating visualizations and plots.
- NLTK (nltk): Natural Language Toolkit, used for natural language processing tasks such as tokenization, stemming, and stopwords removal.
  - nltk.corpus: Provides access to corpora and lexical resources.
  - nltk.tokenize: Used for tokenization of text data.
  - nltk.stem: Provides stemming and lemmatization utilities.
- Scikit-learn (sklearn): Scikit-learn library, used for machine learning tasks
  - sklearn.feature_extraction.text: Provides utilities for converting text data into numerical feature vectors.
  - scipy.sparse: Sparse matrix implementation used in machine learning algorithms.
  - sklearn.model_selection: Provides utilities for splitting datasets and cross-validation.
  - sklearn.linear_model: Implements logistic regression model used for classification.
  - sklearn.metrics: Provides evaluation metrics for assessing model performance.

## Getting Started
To get started with this project, follow these steps:
1. Clone this repository: git clone https://github.com/Kiahmin/Fake-New-Detection.git
2. Open main.ipynb to run the code for preprocessing, model training, and evaluation.
