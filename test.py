from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
np.random.seed(0)


import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt


df = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False)


print("Number of Rows: ", len(df.axes[0]))
print("Number of Columns: ", len(df.axes[1]))
print("-------------------------------------")
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 
                "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                "tcp.dstport", "udp.port", "mqtt.msg"]
df.drop(drop_columns, axis=1, inplace=True)
#remove no values lines
df.dropna(axis=0, how='any', inplace=True)
#remove duplicates
df.drop_duplicates(subset=None, keep="first", inplace=True)
#shuffles the dataset
#df = shuffle(df)
#remove one of the types, Attack_type or Attack_label
#df.pop("Attack_type")
print("Number of Rows: ", len(df.axes[0]))
print("Number of Columns: ", len(df.axes[1]))

categorical_columns = []
categorical_dims =  {}
for col in df.columns[df.dtypes == object]:
    print(col, df[col].nunique())
    l_enc = LabelEncoder()
    df[col] = l_enc.fit_transform(df[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)
    
    
    
#split data
X = df.iloc[:,0:46]
#for binary class
Y_B = df.iloc[:,46]
#for multiclass 
Y = df.iloc[:,47]

# split data into train, test and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=7) # 0.25 x 0.8 = 0.2
print("train X: ", len(X_train))
print("test X: ", len(X_test))
print("val X: ", len(X_val))

clf_xgb = XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective="multi:softmax",
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,
    num_class= len(Y.unique()))
le = LabelEncoder()
y_valid = le.fit_transform(y_val)
clf_xgb.fit(X_train, y_train,
            eval_set=[(X_val, y_valid)],
            early_stopping_rounds=40,
            verbose=10)