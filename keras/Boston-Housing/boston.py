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

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
batch_size = 5
epochs = 50

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=1)

# Test it out!
sample = 0
test = np.expand_dims(x_train[sample], axis=0)
pred = model.predict(test)[0]
print(x_train[sample])
print('predict: ' + str(pred[0]) + ', actual: ' + str(y_train[sample]))
