# real-and-fake-news-classification
This project implements various machine learning models to classify news articles as real or fake. The dataset contains labeled news articles, and the goal is to predict whether a given news article is real or fake based on its content. Experiments were conducted both with and without preprocessing to evaluate the impact of text cleaning and transformation on model performance.
# Project Overview
This project explores the classification of news articles using various machine learning models. The models were trained both on raw text data and preprocessed data to observe the impact of preprocessing techniques on model accuracy.

Key components of this project include:

**Data preprocessing**: Text cleaning, tokenization, and feature extraction.<br/>
**Model training**: Various machine learning models such as Logistic Regression, Random Forest, Support Vector Machines (SVM), and Naive Bayes.<br/>
**Evaluation:** Comparison of performance metrics like accuracy, precision, recall, and F1-score.<br/>
The aim is to find the best combination of model and preprocessing steps for the task of fake news detection.

# Dataset
The dataset used for this project is a collection of news articles, which are labeled as real or fake. You can find a dataset such as the [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) or any other similar dataset.

The dataset contains two columns:

text: The content of the news article.
label: The label indicating whether the article is real (1) or fake (0).
# Preprocessing
In this project, several preprocessing steps were performed on the text data to improve the performance of machine learning models:
<br/>
**Lowercasing:** Convert all text to lowercase.<br/>
**Removing punctuation** and special characters.<br/>
**Tokenization**: Split the text into words or tokens.<br/>
**Stopword Removal:** Remove common words like "the", "is", etc., that don't contribute much meaning.<br/>
**Stemming or Lemmatization:** Reduce words to their root form (e.g., "running" -> "run").<br/>
These steps were applied in separate experiments to evaluate their effect on model performance.
# Experiments
This project includes the following experiments:

## 1. Without Preprocessing
Using raw text data without any preprocessing.<br/>
Train machine learning models and evaluate performance.<br/>
## 2. With Preprocessing
Apply text preprocessing (mentioned above) to clean the data before training.<br/>
Train machine learning models on the preprocessed data.<br/>
## 3. Machine Learning Models Used<br/>
Logistic Regression<br/>
Naive Bayes<br/>
Random Forest<br/>
## 4. Evaluation
Models were evaluated using performance metrics like accuracy, precision, recall, F1-score.
# Results
The experiments revealed that the Multinomial Naive Bayes (MultinomialNB) model, when trained on preprocessed text, outperformed all other models in terms of accuracy and overall performance. The preprocessing steps, which included text normalization, stopword removal, and tokenization, significantly enhanced the model’s ability to distinguish between real and fake news. In particular, the MultinomialNB model with preprocessed text showed a remarkable improvement in accuracy compared to the model trained on raw, unprocessed text, highlighting the importance of text preprocessing in improving model performance for text classification tasks. The enhanced feature representation from preprocessing allowed the model to better capture the underlying patterns in the data, resulting in more accurate predictions and better generalization on the test set.
