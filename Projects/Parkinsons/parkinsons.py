# Written by Nikhil D'Souza
# Data from UCI ML Repo: Parkinsons Data Set (https://archive.ics.uci.edu/ml/datasets/parkinsons)

# This neural network predicts whether subject has Parkinson's disease based on biomedical voice measurements.

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

parkinsons = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')

y = np.array(parkinsons.ix[:,17]) # data from 17th column (Parkinson's status)
parkinsons.drop('status', axis=1, inplace=True)
X = np.array(parkinsons.ix[:,1:24]) # data from all columns except the 17th

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_model():
    # Model architecture
    model = Sequential()
    model.add(Dense(20, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()

# Train model
batch_size = 500
epochs = 5000

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
model.save('model.h5')

# Test it out!
model = load_model('model.h5')
output_labels = ['Parkinsons', 'Healthy']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

print('Prediction: ' + output_labels[int(round(pred))] + ', ' + 'Actual: ' + output_labels[y_train[0]])
