#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

#importing the training dataset
train_data = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

data = train_data['Lemmatized_Text']
#print('\nData Shape', data.shape)
labels = train_data['Category']
#print('\nLabels Shape', labels.shape)

X_train, X_val, y_train, y_val = train_test_split(data, labels,
                                                test_size=0.2, random_state=2, 
                                                stratify = labels)

# print(X_train.shape)
# print(X_val.shape)
# print(y_train.shape)
# print(y_val.shape)

pipe = Pipeline(
    steps = ([
            ("vect", CountVectorizer(min_df=5, max_df=0.7, ngram_range = (1,3))),
            #("vect", TfidfVectorizer(min_df=5, max_df=0.7)),
            ("selector", SelectKBest(score_func=chi2, k=10)),
            ("clf", LogisticRegression(max_iter= 100)),
            #("clf", KNeighborsClassifier(metric = 'manhattan', n_neighbors = 5)),
            #("clf", RandomForestClassifier(class_weight='balanced')),
            #("clf", MultinomialNB(alpha= 16.666733333333333, fit_prior = False)),
            
            ])
    )

param_grid = {
    #KNN
    # 'clf__n_neighbors': [5],
    # 'clf__metric': ['manhattan'],
    
    #Logistic Regression
    'clf__penalty': ['none'],
    'clf__multi_class': ['ovr'],
    'clf__solver': ['saga'],
    
    #Random Forest
    # 'clf__criterion': ['entropy'],
    # 'clf__n_estimators': [200],
    # 'clf__max_depth': [20],
    
    #MultinomialNB
    # 'clf__alpha': [0.0001],
    # 'clf__fit_prior':[False],
}

grid_model = GridSearchCV(pipe, param_grid, verbose=2, cv=10, n_jobs=-1)
grid_model.fit(X_train, y_train)

y_predict = grid_model.predict(X_val)
#print(y_predict.shape)
print('\nClassification Report:\n', classification_report(y_val, y_predict))
print('\nConfusion Matrix:\n', confusion_matrix(y_val, y_predict))
print('\nAccuracy Score:\n', accuracy_score(y_val, y_predict))

#print('\nSaving the best model')
with open('best_model', 'wb') as picklefile:
    pickle.dump(grid_model, picklefile) 

test = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/test_data.csv', names=['Complaint'])
test_data = test['Complaint']
#print('\nTestData Shape', test_data.shape)

#print('\nLoading the best model')
with open('best_model', 'rb') as training_model:
    model = pickle.load(training_model)
    
test_predict = model.predict(test_data)
print('\nPredicted Labels:\n', test_predict)
#print(test_predict.shape)
#np.savetxt("Project6_TestData_ClassLabels.csv", test_predict, delimiter = ',', fmt="%s")
