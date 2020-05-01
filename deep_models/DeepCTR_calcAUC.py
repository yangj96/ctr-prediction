'''
train kinds of deep CTR model, write AUC on test set.
'''
import pandas as pd
import tensorflow as tf
from time import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import *
from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam,Adagrad,RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import KFold
import numpy as np
import argparse
import os
import setproctitle
setproctitle.setproctitle('DeepMF@linian')
parser = argparse.ArgumentParser(description='DeepMF')
parser.add_argument('--batch_size', type=int, default=40960, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='epochs')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate')
parser.add_argument('--embedding_size', type=int, default=8, metavar='N', help='embedding_size')
parser.add_argument('--model', type=str, default='DeepFM', metavar='N', help='model')
parser.add_argument('--val_frac', type=float, default=0.1, metavar='N', help='val_frac')
parser.add_argument('--try_name', type=str, default='', metavar='N', help='try')
parser.add_argument('--gpu', type=str, default='5', metavar='N', help='gpu_id')
parser.add_argument('--k', type=str, default=5, metavar='N', help='k-fold')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
t = time()
train_data_all = pd.read_csv('./train_data_1120.csv')
test_data = pd.read_csv('./test_data_1120.csv')
print('data load done, cost {}s'.format(time()-t))
t = time()

sparse_features = ['poi_id', 'request_cate_id', 'device_type', 'gender', 'job', 'cate_level1', 'cate_level2', 'cate_level3', 'area_id']
# sparse_features = ['poi_id', 'request_cate_id', 'gender', 'cate_level1', 'cate_level2', 'cate_level3', 'area_id']
target = ['action']
dense_features = ['latitude_req', 'longitude_req', 'age', 'avg_price', 'poi_star', 'longitude_poi', 'latitude_poi', 'distance', 'poi_cnt_deal', 'poi_avg_discount', 'request_time_second']

kfold_score = np.zeros(train_data_all.shape[0])
kf = KFold(n_splits = args.k, random_state = 123)
h = 1
for train_index, val_index in kf.split(train_data_all):
    train_data = train_data_all.iloc[train_index]
    val_data = train_data_all.iloc[val_index]

    sparse_feature_columns = [SparseFeat(feat, train_data_all[feat].append(test_data[feat]).nunique()) for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    print('data process done')

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train_model_input = {name:train_data[name].values for name in feature_names}
    test_model_input = {name:test_data[name].values for name in feature_names}
    val_model_input = {name: val_data[name].values for name in feature_names}
    validation_data = ({name: val_data[name].values for name in feature_names}, val_data[target].values)

    model = eval(args.model)
    model = model(linear_feature_columns, dnn_feature_columns,task='binary')
    model.compile(optimizer=RMSprop(args.lr), loss="binary_crossentropy", metrics=['binary_crossentropy'])
    es = EarlyStopping(monitor='val_binary_crossentropy')
    reduce_lr = ReduceLROnPlateau(monitor='val_binary_crossentropy', factor = 0.5, patience=3, mode='auto', min_lr=1e-6)
    print('train now')
    history = model.fit(train_model_input, train_data[target].values, batch_size=args.batch_size, epochs=args.epochs, verbose=2, validation_split=args.val_frac,callbacks=[reduce_lr, es], validation_data=validation_data)
    # save_model(model, '{}_1120_dim{}{}.h5'.format(args.model, args.embedding_size, args.try_name))
    pred_val = model.predict(val_model_input, batch_size=20480)
    print("val AUC", round(roc_auc_score(val_data[target].values, pred_val), 4))
    kfold_score[val_index] = pred_val[:, 0]
    print('{}/{} fold score done\n\n'.format(h, args.k))
    h += 1
print("high AUC", round(roc_auc_score(train_data_all[target].values, kfold_score), 4))
t = pd.DataFrame({'ID': list(range(len(kfold_score))),'action':kfold_score})
t.to_csv("./{}_1120_{}{}.csv".format(args.model, args.k, args.try_name),index=False,sep=',')
print('csv written done')