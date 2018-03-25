# Data from https://en.wikipedia.org/wiki/Iris_flower_data_set

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

import coremltools
import pandas as pd
import numpy as np
import keras

iris = data('iris')

# Data processing
train, test = train_test_split(iris, test_size=0.2)

x = iris.drop('Species', axis=1)
y = iris['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y)

new_x_train = []
for index, item in enumerate(y_train):
    tempArray = []
    tempArray.append(x_train.iloc[index]['Sepal.Length'])
    tempArray.append(x_train.iloc[index]['Sepal.Width'])
    tempArray.append(x_train.iloc[index]['Petal.Length'])
    tempArray.append(x_train.iloc[index]['Petal.Width'])
    new_x_train.append(tempArray)
x_train = new_x_train

new_y_train = []
for index, item in enumerate(y_train):
    new_y_train.append(y_train.iloc[index])
y_train = new_y_train

for index, item in enumerate(y_train):
    if item == 'setosa':
        y_train[index] = 0
    if item == 'versicolor':
        y_train[index] = 1
    if item == 'virginica':
        y_train[index] = 2

new_x_test = []
for index, item in enumerate(y_test):
    tempArray = []
    tempArray.append(x_test.iloc[index]['Sepal.Length'])
    tempArray.append(x_test.iloc[index]['Sepal.Width'])
    tempArray.append(x_test.iloc[index]['Petal.Length'])
    tempArray.append(x_test.iloc[index]['Petal.Width'])
    new_x_test.append(tempArray)
x_test = new_x_test

new_y_test = []
for index, item in enumerate(y_test):
    new_y_test.append(y_test.iloc[index])
y_test = new_y_test

for index, item in enumerate(y_test):
    if item == 'setosa':
        y_test[index] = 0
    if item == 'versicolor':
        y_test[index] = 1
    if item == 'virginica':
        y_test[index] = 2

# Model architecture
model = Sequential()
model.add(Dense(20, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Callback list (saves models at checkpoints)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}.h5',
        monitor='val_loss', save_best_only=True),
]

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train model
batch_size = 5
epochs = 100

model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_data=(np.array(x_test), np.array(y_test)), verbose=1)

# Create CoreML model
coreml_mnist = coremltools.converters.keras.convert(
    'best_model.48.h5',
    input_names=['input'],
    output_names=['output']
)

coreml_mnist.author = 'Nikhil DSouza'
coreml_mnist.license = 'Nikhil'
coreml_mnist.short_description = 'Iris flower data set'

coreml_mnist.save('IrisClassifier.mlmodel')
