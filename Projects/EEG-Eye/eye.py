# Written by Nikhil D'Souza
# Data from UCI ML Repo: EEG Eye State Data Set (https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)

# This neural network predicts whether your eyes are open/closed based on your brainwaves.

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import keras

eye = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff', skiprows=19, header=None)

X = np.array(eye.ix[:,1:13]) # data from first 14 columns
y = np.array(eye.ix[:,14]) # data from 15th column (eye state)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_model():
    # Model architecture - max accuarcy is 55% :(
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()

# Train model
batch_size = 15
epochs = 50

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
model.save('model.h5')

# Test it out!
model = load_model('model.h5')
output_labels = ['open', 'closed']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

print('Prediction: ' + output_labels[int(round(pred))] + ', ' + 'Actual: ' + output_labels[y_train[0]])
