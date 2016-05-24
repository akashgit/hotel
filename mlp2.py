import random
import pandas
import numpy as np
import pickle
from tensorflow.contrib import learn as skflow
from sklearn import metrics, cross_validation
from keras.models import Sequential
from keras.layers import Merge, Dense, Embedding, BatchNormalization, Dropout,Flatten

random.seed(42)

'''Data Loading and Pre-processing'''
data = pandas.read_csv('../data/hotel/data_80.csv')
data = data.convert_objects(convert_numeric=True)
data = data.dropna(axis=0,how='any')
X = data.ix[:, data.columns != 'hotel_cluster']
# X = X.ix[:, X.columns != 'date_time']
# X = X.ix[:, X.columns != 'srch_ci']
# X = X.ix[:, X.columns != 'srch_co']
X = X.ix[:, X.columns != 'orig_destination_distance']
X_cont = data['orig_destination_distance']
y = data['hotel_cluster']

X = np.array(X)[:500000].astype(int)
y = np.array(y)[:500000].astype(int)
X_cont = np.array(X_cont)[:500000].astype(float)

X = np.c_[X, X_cont]
print X.shape
X_train_temp, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)
X_train = [X_train_temp[:,:-1],X_train_temp[:,-1]]
X_test = [X_test[:,:-1],X_test[:,-1]]
print X_train_temp[:,:-1].shape
data=[]

'''Model'''
encoder_a = Sequential()
encoder_a.add(Embedding(np.amax(X)+1,10,input_length=X.shape[1]-1))
encoder_a.add(Flatten())

encoder_b = Sequential()
encoder_b.add(BatchNormalization(input_shape=(1,)))

model = Sequential()
model.add(Merge([encoder_a, encoder_b], mode='concat'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

'''Training criterion, optimizer and evaluation metric'''
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
'''Run the model'''
model.fit(X_train, y_train,
            batch_size=100, nb_epoch=80,
            validation_data=(X_test, y_test))
