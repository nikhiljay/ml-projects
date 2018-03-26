# Written by Nikhil D'Souza
# Data from http://lib.stat.cmu.edu/datasets/boston

# This neural network predicts the values of houses in Boston based on:
# 1. per capita crime rate by town
# 2. proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. proportion of non-retail business acres per town
# 4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. nitric oxides concentration (parts per 10 million)
# 6. average number of rooms per dwelling
# 7. proportion of owner-occupied units built prior to 1940
# 8. weighted distances to five Boston employment centres
# 9. index of accessibility to radial highways
# 10. full-value property-tax rate per $10,000
# 11. pupil-teacher ratio by town
# 12. 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. % lower status of the population

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import coremltools
import keras
import numpy as np

(x_train, y_train), (x_val, y_val) = boston_housing.load_data()

# Model architecture
model = Sequential()
model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Callback list (saves models at checkpoints)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}.h5',
        monitor='val_loss', save_best_only=True),
]

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
batch_size = 5
epochs = 100

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_data=(x_val, y_val), verbose=1)

# Create CoreML model
coreml_mnist = coremltools.converters.keras.convert(
    'best_model.48.h5',
    input_names=['input'],
    output_names=['output']
)

coreml_mnist.author = 'Nikhil DSouza'
coreml_mnist.license = 'Nikhil'
coreml_mnist.short_description = 'Boston housing price regression'

coreml_mnist.save('BostonClassifier.mlmodel')
