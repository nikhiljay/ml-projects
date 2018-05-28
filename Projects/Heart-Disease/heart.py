# Written by Nikhil D'Souza
# Data from UCI ML Repo: Statlog (Heart) Data Set (http://archive.ics.uci.edu/ml/datasets/statlog+(heart))

# This neural network predicts whether subject has cardiovascular disease based on:
# 1. age
# 2. sex
# 3. chest pain type (4 values)
# 4. resting blood pressure
# 5. serum cholestoral in mg/dl
# 6. fasting blood sugar > 120 mg/dl
# 7. resting electrocardiographic results (values 0,1,2)
# 8. maximum heart rate achieved
# 9. exercise induced angina
# 10. oldpeak = ST depression induced by exercise relative to rest
# 11. the slope of the peak exercise ST segment
# 12. number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

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

heart = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', delim_whitespace=True, header=None)

X = np.array(heart.ix[:,1:12]) # data from first 12 columns
y = np.array(heart.ix[:,13]) # array of 13th column (CVD status)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_model():
    # Model architecture
    model = Sequential()
    model.add(Dense(20, input_dim=12, kernel_initializer='normal', activation='relu'))
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
output_labels = ['absent', 'present']
test = np.expand_dims(x_train[0], axis=0)
pred = model.predict(test)[0]

print('Prediction: ' + output_labels[int(round(pred-1))] + ', ' + 'Actual: ' + output_labels[y_train[0]-1])
