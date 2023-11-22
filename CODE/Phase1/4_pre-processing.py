#!/usr/bin/env python3
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

#importing the training dataset
train = pd.read_csv ('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/training_data.csv', 
                     names=['Complaint', 'Category'])

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