# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:18:19 2020

@author: prave
"""


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.utils as ku 
import numpy as np 
import h5py



tokenizer = Tokenizer()

# Data is loaded into the variable 'data'
data = open('C:/Users/prave/Desktop/Projects/Poetry Generation/text_generation/shakespear_poetry.txt').read()

# Here each line in the dataset is converted is considered a seperate entry/input 
corpus = data.lower().split("\n")

# Tokeniser is nothing but each and every word is tokenized (i.e each word is assigned a seperate number) 
tokenizer.fit_on_texts(corpus)

# The total number of words is stored in len
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)


# Keras Model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))  #(# Your Embedding Layer)
model.add(Bidirectional(LSTM(150, return_sequences=True)))  #(# An LSTM Layer)
model.add(Dropout(0.2))  #(# A dropout layer)
model.add(LSTM(100))  #(# Another LSTM Layer)
model.add(Dense(total_words/2, activation='relu'))  #(# A Dense Layer including regularizers)
model.add(Dense(total_words, activation='softmax'))  #(# A Dense Layer)
# Pick an optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')  #(# Pick a loss function and an optimizer)


# Train the model
history = model.fit(predictors, label, epochs=250, verbose=1)


# Save model in the desired directory
model_json = model.to_json()
with open("C:/Users/prave/Desktop/Projects/Poetry Generation/trial-3/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("C:/Users/prave/Desktop/Projects/Poetry Generation/trial-3/model.h5")

