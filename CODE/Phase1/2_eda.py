#!/usr/bin/env python3
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

#importing the training dataset
train = pd.read_csv('/DATA1/NLP/akanksha_19022/akanksha_19022/DATASET/training_data.csv', 
                    names=['Complaint', 'Category'])

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