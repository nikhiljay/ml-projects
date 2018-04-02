# This neural network classifies different types of white blood cells.

from __future__ import print_function
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import keras
import csv
import cv2
import scipy
import operator

num_classes = 5

def get_filename_for_index(index):
    PREFIX = 'images/BloodImage_'
    num_zeros = 5 - len(index)
    path = '0' * num_zeros + index
    return PREFIX + path + '.jpg'

reader = csv.reader(open('./labels.csv'))
next(reader) # skip the header

X = []
y = []

for row in reader:
    label = row[2]
    if len(label) > 0 and label.find(',') == -1:
        filename = get_filename_for_index(row[1])
        img_file = cv2.imread('./' + filename)
        if img_file is not None:
            img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)

X = np.asarray(X)
y = np.asarray(y)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y = np_utils.to_categorical(encoded_y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_model():
    # Model architecture
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120, 160, 3), output_shape=(120, 160, 3)))

    model.add(Conv2D(32, (3, 3), input_shape=(120, 160, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = get_model()

# Train model
batch_size = 32
epochs = 20

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
model.save('model.h5')

# Test it out!
model = load_model('model.h5')
output_labels = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

pred_index, pred_value = max(enumerate(pred), key=operator.itemgetter(1))
actual_index, actual_value = max(enumerate(y_train[0]), key=operator.itemgetter(1))

print('Prediction: ' + output_labels[int(pred_index)] + ', ' + 'Actual: ' + output_labels[int(actual_index)])
print(str(pred_value*100) + '% confidence')
