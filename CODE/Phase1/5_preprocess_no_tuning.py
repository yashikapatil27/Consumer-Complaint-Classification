#!/usr/bin/env python3
#coding: utf-8

import pandas as pd
import nltk
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize

# Getting first 45k rows from pre-processed training_data
trial_45k = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/trial_preprocessed_45k.csv')

# Reading data and target labels
data = trial_45k['Lemmatized_Text']
y = trial_45k['Category']

# Vectorizing data
vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
X = vectorizer.fit_transform(data).toarray()

# Splitting data and target labels into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Running Models on pre-processed data without hyperparameter tuning

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train) 
rf_pred = rf.predict(X_test)
print("--------Random Forest-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test,rf_pred))
print('\nClassification Report:\n', classification_report(y_test,rf_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, rf_pred))

# Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)
print("--------Multinomial NB-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test,mnb_pred))
print('\nClassification Report:\n', classification_report(y_test,mnb_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, mnb_pred))

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print("--------Logistic Regression-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test, log_reg_pred))
print('\nClassification Report:\n', classification_report(y_test, log_reg_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, log_reg_pred))

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("--------KNN-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test, knn_pred))
print('\nClassification Report:\n', classification_report(y_test, knn_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, knn_pred))

#SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("--------SVM-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test, svm_pred))
print('\nClassification Report:\n', classification_report(y_test, svm_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, svm_pred))