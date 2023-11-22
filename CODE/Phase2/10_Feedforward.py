#!/usr/bin/env python3
# coding: utf-8

# # Project 6: Consumer Complaint Classification

# Name of team members: Akanksha Singh (19022)
#                       Archana Yadav (19048)
#                       Yashika Patil (19339)
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

######### Importing libraries
import pandas as pd
import nltk
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models
from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.optimizers import RMSprop
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Embedding 
from keras.layers import Flatten
from keras.layers import Dense

######################################  Importing the training dataset
train_data = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

###################################### Pre-processing

# defining parameters
max_words = 10000
seq_len = 100
embed_dim = 100
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_data['Lowercase_Text'].values)
word_index = tokenizer.word_index

#Tokenisation and padding
X = tokenizer.texts_to_sequences(train_data['Lowercase_Text'].values)
X = pad_sequences(X, maxlen=seq_len)

#Converting categorical labels to numeric
y = pd.get_dummies(train_data['Category']).values

###################################################      Training the model

################## Step 1: Splitting into train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
print("")
print("Train shape: ",X_train.shape, y_train.shape)
print("Test shape: ",X_test.shape, y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 20)
print("Train shape: ",X_train.shape, y_train.shape)
print("Validation shape: ",X_val.shape, y_val.shape)

################## Step 2: Building model
model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))

################### Step 3: Compile
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")

################# Step 4: Model fitting
epochs = 5
batch_size = 32
print("")
print('-------------------- Training --------------------')
ff_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
print("")

################# Step 5: Evaluation
print("")
print('-------------------- Evaluating for test set --------------------')
ff_eval = model.evaluate(X_test, y_test)
print('Test set FF Model\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(ff_eval[0], ff_eval[1]))
print("")
print('-------------------- Predicting for test set --------------------')
y_pred= model.predict(X_test) 
y_pred= np.argmax(y_pred, axis=1)
y_test= np.argmax(y_test, axis=1)
print("")
print(confusion_matrix(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))
model.save('/data3/nlp/akanksha_19022/akanksha_19022/MODEL/FF')

################# Step 6 - Use model to make predictions
print("")
print('-------------------- Predicting for test dataset--------------------')
model = models.load_model('/data3/nlp/akanksha_19022/akanksha_19022/MODEL/FF')
test_data = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/test_data.csv', names=['Complaint'])
print(test_data)

#Tokenisation and padding
seq_len = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(test_data['Complaint'].values)
test_array = tokenizer.texts_to_sequences(test_data['Complaint'].values)
test_array = pad_sequences(test_array, maxlen=seq_len)

label = {0: 'retail_banking', 1: 'credit_reporting', 2: 'mortgages_and_loans', 3: 'debt_collection', 4: 'credit_card'}
predictions = model.predict(test_array)
predictions = np.argmax(predictions, axis=1)
results = []
for i in range(4061):
    results.append(label[predictions[i]])
print('\nPredicted Labels:\n', results)

#Storing the target labels of the test dataset 
np.savetxt("testdata_classlabels_FF.csv", results, delimiter = ',', fmt="%s")