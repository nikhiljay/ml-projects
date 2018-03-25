# Data from http://lib.stat.cmu.edu/datasets/boston

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
