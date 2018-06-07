from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
import keras
import csv
import cv2
import scipy

def get_filename_for_index(index):
    PREFIX = 'flower_photos/Image_'
    return PREFIX + index + '.jpg'

reader = csv.reader(open('./labels.csv'))
next(reader)

X = []
y = []

for row in reader:
    label = row[1]
    if len(label) > 0 and label.find(',') == -1:
        filename = get_filename_for_index(row[0])
        img_file = cv2.imread(filename)
        if img_file is not None:
            img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)

complete_X = np.asarray(X)
complete_y = np.asarray(y)

first_X = complete_X[0:2171]
first_y = complete_y[0:2171]

last_X = complete_X[2172:3669]
last_y = complete_y[2172:3669]

encoder = LabelEncoder()
def encode(y_values):
    encoder.fit(y_values)
    encoded_y = encoder.transform(y_values)
    y_values = np_utils.to_categorical(encoded_y)
    return y_values

complete_y = encode(complete_y)
first_y = encode(first_y)
last_y = encode(last_y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
first_x_train, first_x_test, first_y_train, first_y_test = train_test_split(first_X, first_y, test_size=0.2)
last_x_train, last_x_test, last_y_train, last_y_test = train_test_split(last_X, last_y, test_size=0.2)

feature_layers = [
    Lambda(lambda x: x/127.5 - 1., input_shape=(120, 160, 3), output_shape=(120, 160, 3)),

    Conv2D(32, (3, 3), input_shape=(120, 160, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten()
]

def get_model(num_classes):
    classification_layers = [
        Dense(64),
    	Activation('relu'),
    	Dropout(0.5),
    	Dense(num_classes),
    	Activation('softmax')
    ]

    model = Sequential(feature_layers + classification_layers)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Train model
batch_size = 32
epochs = 20

model = get_model(3) # Daisy, dandelion, and rose
model.fit(first_x_train, first_y_train, batch_size=batch_size, epochs=epochs, validation_data=(first_x_test, first_y_test), verbose=1) # Training affects feature and classification layers

# Freeze feature layers
for l in feature_layers:
	l.trainable = False

model = get_model(2) # Sunflower and tulip
model.fit(last_x_train, last_y_train, batch_size=batch_size, epochs=epochs, validation_data=(last_x_test, last_y_test), verbose=1) # Training will only affect classification layers
