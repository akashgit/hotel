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
data = pandas.read_csv('../data/hotel/test_80.csv')[['srch_destination_id','user_location_country','user_location_region',\
'channel','is_package','is_mobile','srch_rm_cnt','srch_children_cnt','user_location_city','hotel_market','orig_destination_distance'\
,'srch_adults_cnt']]
data = data.convert_objects(convert_numeric=True)
# data = data.dropna(axis=0,how='any')

test = np.array(X).astype(int)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)
data=[]

'''Model'''
model = model_from_json(open('mlp_architecture.json').read())
model.load_weights('mlp_weights.h5')

with open('results.csv','a') as f_handle:
    for x in test:
        np.savetxt(f_handle,np.argmax(model(x))[:5])
