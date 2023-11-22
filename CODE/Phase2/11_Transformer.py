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
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Embedding 
from keras.layers import Flatten
from keras.layers import Dense 
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

###################################### importing the training dataset
train_data = pd.read_csv('/data3/nlp//akanksha_19022/akanksha_19022/DATASET/preprocessed_training_data.csv')

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
################## Step 1: Splitting into train and test
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size = 0.20, random_state = 42)
print("Train shape: ",x_train.shape,y_train.shape)
print("Test shape: ",x_val.shape,y_val.shape)

################## Step 2: Building model
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# vocab_size = 20000  # Only consider the top 20k words
maxlen = 100  # Only consider the first 100 words of each movie review
# embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, max_words, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
#x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
#x = Dropout(0.1)(x)
outputs = Dense(5, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

####################### Compilation
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

########################## Fitting
epochs = 5
batch_size = 32
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val,y_val))
# history = model.fit(x_train, y_train, 
#                     batch_size=32, epochs=5, 
#                     validation_data=(x_val, y_val))

######################### Evalution
results = model.evaluate(x_val, y_val, verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
