#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


#importing the training dataset
train_data = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

data = train_data['Lemmatized_Text']
labels = train_data['Category']

X_train, X_val, Y_train, Y_val = train_test_split(data, labels,
                                                test_size=0.2, random_state=2, 
                                                stratify = labels)

vectorizer_types= ['bow', 'tfidf']

def vectorizer(vectorizer_type):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
    
    if vectorizer_type== 'bow':
        vectorizer = CountVectorizer(min_df=5, max_df=0.7, ngram_range = (1,3))
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
            
    return vectorizer

"""
feature_selector_types = ['chi2', 'mutual_info'] #couldn't resolve errors of discrete and continuous data and class labels
def feature_selector(feature_selector_type):    #for mutual_info hence dropped
    if feature_selector_type == 'chi2':
        return SelectKBest(score_func = chi2)        
    elif feature_selector_type == 'mutual_info':
        return SelectKBest(score_func = mutual_info_classif)
"""
    
parameters=[
    {
        'clf': LogisticRegression(max_iter= 100),
        'clf__penalty': ['l2', 'none'],
        'clf__multi_class': ['ovr', 'multinomial'],
        'clf__C': np.linspace(0.0001, 30, 2),

    },
    {
        'clf': RandomForestClassifier(class_weight='balanced'),
        'clf__criterion': ['gini', 'entropy'],
        'clf__n_estimators': [30, 200],
        'clf__max_depth': [10, 20],
    },
    {
        'clf': MultinomialNB(),
        'clf__alpha': np.linspace(0.0001, 50, 4),
        'clf__fit_prior':[True, False],
    },
    {
        'clf': svm.SVC(max_iter= 10),
        'clf__C': [0.1, 2, 10, 50],
        'clf__kernel': ['linear', 'sigmoid'],
    },
    {
        'clf': KNeighborsClassifier(),
        'clf__n_neighbors': list(range(1, 7, 2)),
        'clf__metric': ['euclidean', 'manhattan', 'minkowski']
    }
]

results = []

for model in parameters: 
    clf = model.pop('clf')
    print(f"\nStarted {str(clf)}")
    print("-------------------------------------------")
    for vectorizer_type in vectorizer_types:
        print(f"\nStarted Vectorizer {str(vectorizer_type)}")
        print("-------------------------------------------")
        
        pipeline = Pipeline([
        ("vector", vectorizer(vectorizer_type)),
        ("select", SelectKBest(score_func = chi2, k = 10)),
        ("clf", clf)])

        print("\nStarted GridSearchCV")
        print("-------------------------------------------")
        grid_model = GridSearchCV(pipeline, model, verbose=2, cv=10
                                  , scoring='accuracy', error_score='raise')
        grid_model.fit(X_train, Y_train)

        print("Done")
        print(f"Training Score: {grid_model.best_score_}")
        print(f"Parameters: {grid_model.best_params_}") 
        print(f"Best Classifier: {grid_model.best_estimator_}")
        
        results.append({
            'Model': clf,
            'Best_Score': grid_model.best_score_,
            'Best_Params': grid_model.best_params_
        })
        
print(results)
classification_results= pd.DataFrame(results, columns=['Model', 'Best_Score', 'Best_Params'])
np.savetxt("Classification_Results.csv", classification_results, delimiter = ',')
print(classification_results)
