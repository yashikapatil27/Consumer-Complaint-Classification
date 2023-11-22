#!/usr/bin/env python3
# coding: utf-8

# # Project 6: Consumer Complaint Classification

# Name of team members: Akanksha Singh (19022)
#                       Archana Yadav (19048)
#                       Yashika Patil (19339)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

#importing the training dataset
train = pd.read_csv('/data3/nlp/akanksha_19022/akanksha_19022/DATASET/training_data.csv', names=['Complaint', 'Category'])


"""
DATA CLEANING
"""


#removing the NaN/NULL value
train.dropna(how='any',axis=0, inplace = True)

#defining funcction to clean dataset
def text_cleaning(text):
    text = [re.sub(r'@\S+', '', t) for t in text ]
    text = [re.sub(r'#', '', t) for t in text ]
    text = [re.sub(r"https?\S+", '', t) for t in text ]
    text = [re.sub(r"\d*", '', t) for t in text ]    
    text = [re.sub(r"[+|-|*|%]", '', t) for t in text ]  
    text = [re.sub(r"[^^(éèêùçà)\x20-\x7E]", '', t) for t in text]
    return text

train['Cleaned_Text'] = text_cleaning(train['Complaint'])


"""
EXPLORATORY DATA ANALYSIS
"""


grouped_by_class = train.groupby('Category').count()

category = ['retail_banking', 'credit_reporting', 'mortgages_and_loans','debt_collection', 'credit_card']
complaint = [15177, 88892, 22569, 18515, 13197]

# creating the bar plot
fig = plt.figure(figsize = (9, 4))
plt.bar(category, complaint, width=0.4)
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.title("Label Frequency")
plt.show()

# creating the pie chart
fig = plt.figure(figsize = (7, 6))
plt.pie(complaint,labels=category, startangle=90, shadow=True,explode=(0.1, 0.1, 0.1, 0.1,0.1), autopct='%1.2f%%')
plt.title("Label Frequency")
plt.axis('equal')
plt.show()

# making word cloud
for product_name in train['Category'].unique():
    print("\n Category: ", product_name)
    print("\n")

    all_words =''.join([text for text in train.loc[train['Category'].str.contains(product_name)]])

    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()



"""
NO PRE-PROCESSING NO HYPERPARAMETER TUNING
"""


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

# Getting first 45k rows from pre-processed training data
trial_45k = pd.read_csv('/data3/nlp/akanksha_19022/akanksha_19022/DATASET/trial_preprocessed_45k.csv')

# Reading data and target labels
data = trial_45k['Complaint']   #RAW DATA
y = trial_45k['Category']

# Vectorizing data
vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
X = vectorizer.fit_transform(data).toarray()

# Splitting data and target labels into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Running Models on raw data without hyperparameter tuning

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train) 
rf_pred = rf.predict(X_test)
print("--------Random Forest-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test,rf_pred))
print('\nClassification Report:\n', classification_report(y_test,rf_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, rf_pred))

#Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)
print("--------Multinomial NB-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test,mnb_pred))
print('\nClassification Report:\n', classification_report(y_test,mnb_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, mnb_pred))

#Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print("--------Logistic Regression-------")
print('\nConfusion Matrix:\n', confusion_matrix(y_test, log_reg_pred))
print('\nClassification Report:\n', classification_report(y_test, log_reg_pred))
print('\nAccuracy Score:\n', accuracy_score(y_test, log_reg_pred))

#KNN
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
 

    
"""
PRE-PROCESSING
"""
    

import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet') 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree = "".join([i for i in str(text) if i not in string.punctuation])
    return punctuationfree
train['Unpunct_Text']= train['Cleaned_Text'].apply(lambda x:remove_punctuation(x))

#changing to lower case
train['Lowercase_Text']= train['Unpunct_Text'].apply(lambda x: x.lower())

#defining function for tokenization
def tokenization(text):
    tokens = sent_tokenize(text)
    return tokens
train['Tokenised_Text']=train['Lowercase_Text'].apply(lambda x: tokenization(x))

#defining the function to remove stopwords from tokenized text
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
train['No_Stopwords'] = train['Tokenised_Text'].apply(lambda x: remove_stopwords(x))

# defining a function for stemming
porter_stemmer = PorterStemmer()
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
train['Stemmed_Text']=train['No_Stopwords'].apply(lambda x: stemming(x))

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
train['Lemmatized_Text']=train['Stemmed_Text'].apply(lambda x:lemmatizer(x))   



"""
PRE-PROCESSED DATA BUT NO HYPERPARAMETER TUNING
"""


# Getting first 45k rows from pre-processed training_data
trial_45k = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/trial_preprocessed_45k.csv')

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



"""
HYPERPARAMETER TUNING ON PRE-PROCESSED DATA
"""


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
train_data = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

data = train_data['Lemmatized_Text']
labels = train_data['Category']

# splitting training and validation data and target variables
X_train, X_val, Y_train, Y_val = train_test_split(data, labels,
                                                test_size=0.2, random_state=2, 
                                                stratify = labels)

# selecting vectorizer for data manipulation
vectorizer_types= ['bow', 'tfidf']

def vectorizer(vectorizer_type):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
    
    if vectorizer_type== 'bow':
        vectorizer = CountVectorizer(min_df=5, max_df=0.7, ngram_range = (1,3))
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
            
    return vectorizer

# mutual info gain threw error with hyperparamters of few models, 
# couldn't resolve issues of discrete/continuous data/label.
# feature selection models
"""
feature_selector_types = ['chi2', 'mutual_info'] #couldn't resolve errors of discrete and continuous data and class labels
def feature_selector(feature_selector_type):    #for mutual_info hence dropped
    if feature_selector_type == 'chi2':
        return SelectKBest(score_func = chi2)        
    elif feature_selector_type == 'mutual_info':
        return SelectKBest(score_func = mutual_info_classif)
"""

# pipeline parameter grid    
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

# storing the best hyperparameters for all the combinations
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
        ("select", SelectKBest(score_func = chi2, k =10)),
        ("clf", clf)])

        print("\nStarted GridSearchCV")
        print("-------------------------------------------")
        grid_model = GridSearchCV(pipeline, model, verbose=2, cv=5
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

# saving the best classification results of the all the combinations
classification_results= pd.DataFrame(results, columns=['Model', 'Best_Score', 'Best_Params'])
np.savetxt("Classification_Results.csv", classification_results, delimiter = ',')
print(classification_results)  



"""
TEST DATA PREDICTION
"""


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
train_data = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

data = train_data['Lemmatized_Text']
labels = train_data['Category']

# splitting training and validation data and target variable
X_train, X_val, y_train, y_val = train_test_split(data, labels,
                                                test_size=0.2, random_state=2, 
                                                stratify = labels)

# classification pipeline
# logistic regression had the best accuracy out of all the
# best hyperparameter set of the each classifier
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

# parameter grid
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

# fitting the model
grid_model = GridSearchCV(pipe, param_grid, verbose=2, cv=5, n_jobs=-1)
grid_model.fit(X_train, y_train)

# predicting the validation set target labels
y_predict = grid_model.predict(X_val)
print('\nClassification Report:\n', classification_report(y_val, y_predict))
print('\nConfusion Matrix:\n', confusion_matrix(y_val, y_predict))
print('\nAccuracy Score:\n', accuracy_score(y_val, y_predict))

# saving the best model
with open('best_model', 'wb') as picklefile:
    pickle.dump(grid_model, picklefile) 

# reading the test dataset
test = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/test_data.csv', names=['Complaint'])
test_data = test['Complaint']

#loading the best model
with open('best_model', 'rb') as training_model:
    model = pickle.load(training_model)
 
# predicting the target labels of the test dataset    
test_predict = model.predict(test_data)
print('\nPredicted Labels:\n', test_predict)

# storing the target labels of the test dataset 
np.savetxt("Project6_TestData_ClassLabels.csv", test_predict, delimiter = ',', fmt="%s")
