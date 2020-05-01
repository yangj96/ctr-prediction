#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

input_train = [] 
input_test = []

def add_base_train(model_name):
    model_oof_train = pd.read_csv(os.path.join('oof', model_name))
    print(model_oof_train.shape)
    input_train.append(model_oof_train)

    
def add_base_test(model_name):
    model_oof_test = pd.read_csv(os.path.join('oof', model_name))
    print(model_oof_test.shape)
    input_test.append(model_oof_test)

train_dir = ('./train_result')
test_dir = ('./test_result')
train_oofs = os.listdir(train_dir)
train_oofs = [x for x in train_oofs if not x.startswith(".")]

test_oofs = os.listdir(train_dir)
test_oofs = [x for x in train_oofs if not x.startswith(".")]

for i in range(len(train_oofs)):
    add_base_train(train_oofs[i])
    add_base_test(test_oofs[i])


stacked_train = np.concatenate([f.action.values.reshape(-1, 1) for f in input_train], axis=1)
stacked_test = np.concatenate([f.action.values.reshape(-1, 1) for f in input_test], axis=1)

train_df = pd.read_csv('data/train_data_1120.csv')
test_df = pd.read_csv('data/test_data_1120.csv')

train_target = train_df['action']
test_id = test_df['Unnamed: 0']

# second layer
n_splits = 5
random_state = 2000
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(stacked_train, train_target))

oof = np.zeros(len(stacked_train))
predictions = np.zeros(len(stacked_test))

for i, (train_idx, valid_idx) in enumerate(splits):
    print('Folder', i)
    x_tr, y_tr = stacked_train[train_idx], train_target.iloc[train_idx]
    x_valid, y_valid = stacked_train[valid_idx], train_target.iloc[valid_idx]
    
    clf = LinearRegression().fit(x_tr, y_tr)

    oof[valid_idx] = clf.predict(x_valid)
    predictions += clf.predict(stacked_test) / n_splits
    
    del x_tr
    del y_tr

print(metrics.roc_auc_score(train_target.values, oof))


submission = pd.DataFrame({'ID': test_id, 'action': predictions})
submission.to_csv('result/stacking.csv', index = False)