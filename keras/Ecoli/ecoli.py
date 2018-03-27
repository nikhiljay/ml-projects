# Written by Nikhil D'Souza
# Data from UCI ML Repo: Ecoli Data Set (http://archive.ics.uci.edu/ml/datasets/Ecoli)

# This neural network predicts the localization site of a protein based on:
# 1. Accession number for the SWISS-PROT database
# 2. McGeoch's method for signal sequence recognition
# 3. von Heijne's method for signal sequence recognition
# 4. von Heijne's Signal Peptidase II consensus sequence score
# 5. Presence of charge on N-terminus of predicted lipoproteins
# 6. score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins
# 7. score of the ALOM membrane spanning region prediction program
# 8. score of ALOM program after excluding putative cleavable signal regions from the sequence

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import keras

ecoli = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', delim_whitespace=True, header=None)

X = np.array(ecoli.ix[:,1:7]) # data from first 7 columns
y = np.array(ecoli.ix[:,8]) # array of 8th column (localization sites)
target = LabelEncoder().fit_transform(y) # y converted into numbers

x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

# Model architecture
model = Sequential()
model.add(Dense(1000, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train model
batch_size = 15
epochs = 500

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

# Test it out!
output_labels = ['cp', 'im', 'imS', 'imL', 'imU', 'om', 'omL', 'pp']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

print('Prediction: ' + output_labels[int(round(pred))] + ', ' + 'Actual: ' + output_labels[y_train[0]])
