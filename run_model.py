import random
import pandas
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Merge, Dense, Embedding, BatchNormalization, Dropout,Flatten

random.seed(42)

'''Data Loading and Pre-processing'''
data = pandas.read_csv('../data/hotel/test.csv')[['srch_destination_id','user_location_country','user_location_region',\
'channel','is_package','is_mobile','srch_rm_cnt','srch_children_cnt','user_location_city','hotel_market'\
,'srch_adults_cnt']]
data = data.convert_objects(convert_numeric=True)
# data = data.dropna(axis=0,how='any')

test = np.array(data).astype(int)
test=np.clip(test,0,65776)
#data=[]

'''Model'''
model = model_from_json(open('mlp_without_dist.json').read())
model.load_weights('mlp_without_dist_weights.h5')

model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])



i=0
with open('results.csv','a') as f_handle:
    for x in test:
        try:
            f_handle.write(str(np.argsort(-model.predict_on_batch(x.reshape(1,11))).astype(int)[0][:5]))
            f_handle.write('\n')

        except:
            print i
            try:
                f_handle.write(str(np.argsort(-model.predict_on_batch(test[i].reshape(1,11))).astype(int)[0][:5]))
                f_handle.write('\n')
            except:
                print i
                print 'okay this time it really fucked up'
                continue
        i+=1
