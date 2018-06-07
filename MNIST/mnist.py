# Written by Nikhil D'Souza
# Data from http://yann.lecun.com/exdb/mnist/

# This neural network predicts what number is written in an image.

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras import backend as K

import coremltools
import keras
import cv2
import numpy as np
import operator

# x is image and y is answer
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# plt.imshow(x_train[0])
# plt.show()

# Numbers of rows and coloumns = 28
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
num_classes = 10  # one class for each number

# Reshape images and change to black and white (channel 1)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Keras needs type float32 instead of type uint8
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

# Change MNIST data values to between 0 and 1
x_train /= 255
x_val /= 255

# Convert y values from 1 dimensional array into 10 dimensional array
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

# Model architecture
model = Sequential()

# First layer
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu')) # Parameters = # of filter/features, layer size, activation function
model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling filter halves width and height reducing # of parameters (helps control overfitting)
model.add(Dropout(0.5)) # Randomly sets a fraction of inputs to 0 (works with pooling to help control overfitting)

# Second layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Third layer
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) # Make weights from layers 1 dimensional
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Train model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 200
epochs = 1

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=1)

# Test out the model!
image = cv2.imread("test2.png", 0)
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

pred = model.predict(image)[0]
index, value = max(enumerate(pred), key=operator.itemgetter(1))
print(str(index) + ', ' + str(value*100) + '% confidence')
