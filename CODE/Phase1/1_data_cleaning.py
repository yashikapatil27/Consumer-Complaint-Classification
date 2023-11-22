#!/usr/bin/env python3
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

#importing the training dataset
train = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/training_data.csv', names=['Complaint', 'Category'])

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