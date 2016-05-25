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
data = pandas.read_csv('../data/hotel/data_80.csv')[['user_location_country','user_location_region','user_location_city','hotel_market','orig_destination_distance']]
data = data.convert_objects(convert_numeric=True)
data = data.dropna(axis=0,how='any')
X = data.ix[:, data.columns != 'orig_destination_distance']
y = data['orig_destination_distance']

X = np.array(X)[:500000].astype(int)
y = np.array(y)[:500000].astype(int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)
data=[]
print len(np.unique(y))

'''Model'''
model = Sequential()
model.add(Embedding(np.amax(X)+1,100,input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y)), activation='softmax'))

'''Training criterion, optimizer and evaluation metric'''
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
'''Run the model'''
model.fit(X_train, y_train,
            batch_size=100, nb_epoch=20,
            validation_data=(X_test, y_test))
