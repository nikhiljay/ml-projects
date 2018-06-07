# Written by Nikhil D'Souza
# Data from https://en.wikipedia.org/wiki/Iris_flower_data_set

# This neural network predicts the specific species of an Iris flower based on:
# 1. Sepal length
# 2. Sepal width
# 3. Petal length
# 4. Petal width

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

import pandas as pd
import numpy as np
import keras

iris = datasets.load_iris()

X = iris.data[:, :4] #
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model architecture
model = Sequential()
model.add(Dense(1000, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

# Train model
batch_size = 5
epochs = 500

model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=epochs, validation_data=(np.array(x_test), np.array(y_test)), verbose=1)

# Test it out!
output_labels = ['setosa', 'versicolor', 'virginica']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

print('Prediction: ' + output_labels[int(round(pred))] + ', ' + 'Actual: ' + output_labels[y_train[0]])
