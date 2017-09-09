'''
Created on Aug 1, 2017

@author: aman.dalmia
'''

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

train = pd.read_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/inputData/train.csv')
test = pd.read_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/inputData/test.csv')

#print ('The train inputData has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
#print ('The test inputData has {} rows and {} columns'.format(test.shape[0],test.shape[1]))

# imputing missing values
train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None",inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None",inplace=True)
test['devid'].fillna("None",inplace=True)


# create timebased features

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['tweekday'] = train['datetime'].dt.weekday
test['tweekday'] = test['datetime'].dt.weekday

train['thour'] = train['datetime'].dt.hour
test['thour'] = test['datetime'].dt.hour

train['tminute'] = train['datetime'].dt.minute
test['tminute'] = test['datetime'].dt.minute

# create aggregate features
site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_offer_count_test = test.groupby(['siteid','offerid']).size().reset_index()
site_offer_count_test.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_cat_count_test = test.groupby(['siteid','category']).size().reset_index()
site_cat_count_test.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

site_mcht_count_test = test.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count_test.columns = ['siteid','merchant','site_mcht_count']

# joining all files
agg_df = [site_offer_count,site_cat_count,site_mcht_count]
agg_df_test = [site_offer_count_test,site_cat_count_test,site_mcht_count_test]

for x in agg_df:
    train = train.merge(x)
    
for x in agg_df_test:
    test = test.merge(x)
    
# Label Encoding
from sklearn.preprocessing import LabelEncoder
for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
        
# sample 10% inputData - to avoid memory troubles
# if you have access to large machines, you can use more inputData for training

#train = train.sample(int(1e6))
#print (train.shape)

# select columns to choose
cols_to_use = [x for x in train.columns if x not in list(['ID','datetime','click'])]


# standarise inputData before training
scaler = StandardScaler().fit(train[cols_to_use])

strain = scaler.transform(train[cols_to_use])
stest = scaler.transform(test[cols_to_use])

# train validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.25, random_state=2017)

# print (X_train.shape)
# print (X_valid.shape)
# print (Y_train.shape)
# print (Y_valid.shape)

# model architechture
def keras_model(train):
    
    input_dim = train.shape[1]
    classes = 2
    
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape = (input_dim,)))
    model.add(Dense(60, activation = 'relu'))
    model.add(Dense(classes, activation = 'sigmoid'))
    model.compile(optimizer = 'sgd', loss='binary_crossentropy',metrics = ['accuracy'])
    return model

callback = EarlyStopping(monitor='val_acc',patience=3)

# one hot target columns
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)

# train model
model = keras_model(X_train)
model.fit(X_train, Y_train, 10000, 50, callbacks=[callback],validation_data=(X_valid, Y_valid),shuffle=True)

# check validation accuracy
vpreds = model.predict_proba(X_valid)[:,1]
score = roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)
print("###############################################")
print(score)
print("###########################################")

# predict on test inputData
test_preds = model.predict_proba(stest)[:,1]

# create submission file
submit = pd.DataFrame({'ID':test.ID, 'click':test_preds})
submit.to_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/outputData/keras_starter.csv', index=False)

