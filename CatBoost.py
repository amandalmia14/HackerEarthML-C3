'''
Created on Aug 1, 2017

@author: aman.dalmia
'''

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/inputData/train.csv')
test = pd.read_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/inputData/test.csv')

train.isnull().sum(axis=0)/train.shape[0]

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)

# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
    
cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

# catboost accepts categorical variables as indexes
cat_cols = [0,1,2,4,6,7,8]

# modeling on sampled (1e6) rows
# rows = np.random.choice(train.index.values, 1e6)
# sampled_train = train.loc[rows]


trainX = train[cols_to_use]
trainY = train['click']

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.25)
model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.01, eval_metric='AUC', random_seed=1)


model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

pred = model.predict_proba(test[cols_to_use])[:,1]

sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('D:/Personal/athon/WorkspaceML/HackerEarth-ML-Challenge-3/outputData/cb_sub1.csv',index=False)
