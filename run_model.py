import random
import pandas
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Merge, Dense, Embedding, BatchNormalization, Dropout,Flatten

random.seed(42)

'''Data Loading and Pre-processing'''
data = pandas.read_csv('../data/hotel/test_80.csv')[['srch_destination_id','user_location_country','user_location_region',\
'channel','is_package','is_mobile','srch_rm_cnt','srch_children_cnt','user_location_city','hotel_market'\
,'srch_adults_cnt']]
data = data.convert_objects(convert_numeric=True)
# data = data.dropna(axis=0,how='any')

test = np.array(data).astype(int)

#data=[]

'''Model'''
model = model_from_json(open('mlp_without_dist.json').read())
model.load_weights('mlp_without_dist_weights.h5')

model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

test=np.clip(test,np.amin(test),65776)

i=0
with open('results.csv','a') as f_handle:
    while i <= range(len(test)):
        np.savetxt(f_handle,model.predict_on_batch(test[i:i+100]))
	i=i+100
