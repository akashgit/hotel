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
data = pandas.read_csv('../data/hotel/data_80.csv')[['srch_destination_id','user_location_country','user_location_region',\
'channel','is_package','is_mobile','srch_rm_cnt','srch_children_cnt','user_location_city','hotel_market','orig_destination_distance'\
,'srch_adults_cnt','hotel_cluster']]
data = data.convert_objects(convert_numeric=True)
data = data.dropna(axis=0,how='any')
X = data.ix[:, data.columns != 'hotel_cluster']
y = data['hotel_cluster']

X = np.array(X)[:500000].astype(int)
y = np.array(y)[:500000].astype(int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)
data=[]

'''Model'''
model = Sequential()
model.add(Embedding(np.amax(X)+1,100,input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

'''Training criterion, optimizer and evaluation metric'''
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
'''Run the model'''
model.fit(X_train, y_train,
            batch_size=100, nb_epoch=40,
            validation_data=(X_test, y_test))

json_string = model.to_json()
open('mlp.json', 'w').write(json_string)
model.save_weights('mlp_weights.h5')
