# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:38:37 2024

@author: yash.garg
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Download stopwords if not already available
nltk.download('stopwords')

# Load dataset
news_df = pd.read_csv('D:/OneDrive - CAMPUS ACTIVEWEAR LIMITED/Data/Desktop/Daily report/2024/Work/Project/datasets/train.csv')

# Fill missing values
news_df = news_df.fillna(' ')

# Create 'content' column by combining 'author' and 'title'
news_df['content'] = news_df['author'] + " " + news_df['title']

# Initialize the PorterStemmer
ps = PorterStemmer()

# Stemming function
def stemming(content):
    # Remove non-alphabetic characters and convert to lowercase
    stemmed_content = re.sub(r'[^a-zA-Z\s]', '', content)
    stemmed_content = stemmed_content.lower()
    
    # Split into words and stem
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    
    # Join the list of stemmed words into a single string
    return " ".join(stemmed_content)

# Apply stemming to the content column
news_df['content'] = news_df['content'].apply(stemming)

# Split data into input features and target labels
X = news_df['content'].values
Y = news_df['label'].values

# Convert text data to numerical using TF-IDF vectorizer
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model on the training data
train_Y_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_Y_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Prediction function
def predict_news(text):
    # Preprocess the input text by applying stemming
    processed_text = stemming(text)
    
    # Vectorize the input text
    vectorized_text = vector.transform([processed_text])
    
    # Predict using the trained model
    prediction = model.predict(vectorized_text)
    
    return prediction[0]

# Streamlit interface
st.title('Fake News Detector')

# Input text from the user
input_text = st.text_input('Enter news article:')

if input_text:
    # Make prediction
    prediction = predict_news(input_text)
    
    # Display the result
    if prediction == 1:
        st.write("The news article is likely **FAKE**.")
    else:
        st.write("The news article is likely **REAL**.")
