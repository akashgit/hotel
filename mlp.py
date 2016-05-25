import random
import pandas
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn as skflow
from sklearn import metrics, cross_validation
from keras.models import Sequential
from keras.layers import Merge, Dense, Embedding, BatchNormalization, Dropout

random.seed(42)

'''Data Loading and Pre-processing'''
data = pandas.read_csv('../data/hotel/data_80.csv')
X = data.ix[:, data.columns != 'hotel_cluster']
y = data['hotel_cluster']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

X_cat = X.ix[:, X.columns != 'orig_destination_distance']
X_cont = X['orig_destination_distance']

X_test_cat = X_test.ix[:, X_test.columns != 'orig_destination_distance']
X_test_cont = X_test['orig_destination_distance']

X_train = [X_cat,X_cont]
X_test = [X_test_cat,X_test_cont]

encoder_a = Sequential()
encoder_a.add(Embedding(np.amax(X_cat), 200,input_length=X_cat.shape[1]))

encoder_b = Sequential()
encoder_b.add(BatchNormalization(input_shape=(1,)))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
decoder.add(Dense(150, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(100, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(100, activation='softmax'))

decoder.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

decoder.fit(X_train, y_train,
            batch_size=1000, nb_epoch=10,
            validation_data=(x_test, y_test))
