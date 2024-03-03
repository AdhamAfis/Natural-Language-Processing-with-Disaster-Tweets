# Text Classification with TF-IDF and BERT

This repository contains code for text classification using two different approaches: TF-IDF with Logistic Regression and BERT (Bidirectional Encoder Representations from Transformers). The project aims to compare the performance of these methods on the task of classifying tweets as either referring to real disasters or not.

## Introduction
Text classification is a fundamental task in natural language processing (NLP) where the goal is to categorize text documents into predefined classes or categories. In this project, we explore two popular methods for text classification: traditional machine learning with TF-IDF and deep learning with BERT.

## Dataset
We used the "Real or Not? NLP with Disaster Tweets" dataset from Kaggle, which contains tweets labeled as either referring to real disasters or not. The dataset includes text features such as the tweet content, keywords, and location. The dataset is split into training and testing sets, with the majority of tweets labeled for training and a subset reserved for evaluation.

## Data Exploration
We performed exploratory data analysis (EDA) on the dataset to gain insights into the distribution of target labels (real disaster or not) and the frequency of keywords in the tweets. Visualizations such as bar plots and word clouds were used to illustrate these insights.

## Model Building
### TF-IDF and Logistic Regression
We first built a baseline model using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization combined with Logistic Regression. This approach converts text documents into numerical features and trains a logistic regression classifier. We experimented with different preprocessing techniques, such as tokenization, stop word removal, and stemming, to optimize the TF-IDF representation.

### BERT (Bidirectional Encoder Representations from Transformers)
Next, we utilized BERT, a state-of-the-art pre-trained language model, for text classification. BERT captures contextual information from text by considering the entire input sequence bidirectionally. We fine-tuned the pre-trained BERT model on our tweet classification task by adding a classification layer on top and training it on the labeled tweet data. Hyperparameter tuning and learning rate scheduling were employed to optimize the BERT model's performance.

## Results
### TF-IDF Model
- Accuracy: 78.86%
- Precision: 78.84%
- Recall: 68.88%
- F1 Score: 73.52%

### BERT Model
- Accuracy: 84.24%
- Precision: 83.91%
- Recall: 77.97%
- F1 Score: 80.83%

## Conclusion
The BERT model outperformed the TF-IDF baseline, achieving higher accuracy and F1-score. BERT's ability to capture contextual information led to better performance on the text classification task. Further experimentation and fine-tuning of hyperparameters could potentially improve the performance of both models. Insights gained from this project can be applied to various NLP tasks, such as sentiment analysis, document classification, and entity recognition.

## Future Work
- Experiment with different pre-trained language models, such as RoBERTa, GPT, and XLNet, to further improve performance.
- Explore ensemble methods to combine predictions from multiple models for better classification accuracy.
- Investigate advanced text preprocessing techniques and feature engineering methods to enhance model performance.
- Deploy the trained models as part of a real-time text classification system for disaster response and monitoring.

## Instructions for Use
1. Clone this repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the notebooks `TF_IDF_Baseline.ipynb` and `BERT_Model.ipynb` to replicate the experiments.
4. Explore the code and experiment with different parameters to customize the models.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

