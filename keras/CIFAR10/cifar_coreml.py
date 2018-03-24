from __future__ import print_function
from matplotlib import pyplot as plt
from keras.datasets import cifar10
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

output_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# x is image and y is outcome
(x_train, y_train), (x_val, y_val) = cifar10.load_data()

# plt.imshow(x_train[1])
# plt.show()

# Numbers of rows and coloumns = 32
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
num_classes = 10 # one class for each number

# Reshape images and change to rgb (channel 3)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

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
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Callback list (saves models at checkpoints)
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1) # stop early if accuracy doesnt improve for two consecutive epochs
]

# Train model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 200
epochs = 9

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, validation_data=(x_val, y_val), verbose=1)

# Create CoreML model
coreml_mnist = coremltools.converters.keras.convert(
    'best_model.09.h5', input_names=['image'], output_names=['output'],
    class_labels=output_labels, image_input_names='image')

coreml_mnist.author = 'Nikhil DSouza'
coreml_mnist.license = 'Nikhil'
coreml_mnist.short_description = 'Image based recognition (CIFAR)'
coreml_mnist.input_description['image'] = 'Color image'
coreml_mnist.output_description['output'] = 'Probability of each picture'
coreml_mnist.output_description['classLabel'] = 'Labels of picture'

coreml_mnist.save('CIFARClassifier.mlmodel')
