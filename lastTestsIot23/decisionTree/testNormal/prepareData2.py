import pandas as pd
import io
import requests
import numpy as np
import os
import seaborn as sns

from scipy.stats import zscore

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from matplotlib import pyplot
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from IPython.display import display

from imblearn.over_sampling import SMOTE

import tensorflow as tf

def count_upper_string(string):
    upper = 0
    for c in string:
        if not c.islower():
            upper += 1
    return upper

def count_lower_string(string):
    lower = 0
    for c in string:
        if c.islower():
            lower += 1
    return lower

def one_hot_encoding(df, columns):
    for col in columns:
        print(f'[ONE HOT ENCONDING] {col}')
        df = pd.get_dummies(df, columns=[col], prefix=col)
    return df

def missed_bytes(missed_bytes):
    if missed_bytes < 1:
        return 0
    else:
        return 1
    
def remove_outliers(df,columns,n_std):
    for col in columns:
        print(f'[REMOVE OUTLIERS] {col}')
        
        mean = df[col].mean()
        sd = df[col].std()
        
        df = df[(df[col] <= mean+(n_std*sd))]
        
    return df

def zscore_normalization(df, cols):
    # Standardize the selected columns
    for col in cols:
        if col not in df.columns:
            print(f"[WARNING] {col} not found in DataFrame.")
            continue
        df[col] = zscore(df[col])
    
    print("[DONE] Z-score Normalization")
    print("[INFO] Current Fields in the DataFrame:")
    return df

def delete_columns(df, cols):
    for col in cols:
        df.drop(col, axis = 1, inplace = True)
        print(f'[REMOVED] {col}')
    
    return df


df = pd.read_csv("../../../data/2022_combined_new.csv")

# Remove rows with '-' character in columns 7, 8 and 9
cols_to_check = ['duration', 'orig_bytes', 'resp_bytes']
#cols_to_check = ['duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'history', 'flow_duration']

mask = df[cols_to_check].apply(lambda x: x.str.contains('-', na=False)).any(axis=1)
df = df[~mask]

# Replace comma with period as decimal separator

cols_to_float = ['duration']
df[cols_to_float] = df[cols_to_float].replace(',', '.', regex=True)

# Convert columns 7, 8, 9, and 17 to float and int data type
cols_to_int = ['orig_bytes', 'resp_bytes']

df[cols_to_float] = df[cols_to_float].astype(float)
df[cols_to_int] = df[cols_to_int].astype(int)


df['history_originator'] = df.apply(lambda row: count_upper_string(row['history']), axis=1)
df['history_responder'] = df.apply(lambda row: count_lower_string(row['history']), axis=1)

cols_to_encode = [
    'proto',
    'conn_state',
    'fwd_header_size_min',
    'fwd_header_size_max',
    'bwd_header_size_min',
    'bwd_header_size_max',
    'flow_FIN_flag_count',
    'flow_SYN_flag_count',
    'flow_RST_flag_count',
    'history_originator',
    'history_responder',
]

df = one_hot_encoding(df,cols_to_encode)

df['missed_bytes'] = df.apply(lambda row: missed_bytes(row['missed_bytes']), axis=1)

outliers = [
    'orig_pkts',
    'resp_pkts',
    'orig_ip_bytes',
    'resp_ip_bytes',
]

df = remove_outliers(df, outliers, 3)


cols_to_zscore = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot',
    'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec',
    'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot', 
    'bwd_header_size_tot', 'fwd_PSH_flag_count',
    'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_pkts_payload.min',
    'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg',
    'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max',
    'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg', 'bwd_pkts_payload.std',
    'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot',
    'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
    'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std',
    'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std',
    'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg',
    'flow_iat.std', 'payload_bytes_per_second', 'fwd_subflow_pkts',
    'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes',
    'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets',
    'fwd_bulk_rate', 'bwd_bulk_rate', 'active.max', 'active.tot',
    'active.avg', 'active.std', 'idle.min', 'idle.max', 'idle.tot',
    'idle.avg', 'idle.std', 'fwd_init_window_size', 'bwd_init_window_size',
    'fwd_last_window_size', 'bwd_last_window_size', 'duration', 'orig_bytes',
    'resp_bytes', 'orig_pkts', 'resp_pkts', 'resp_ip_bytes', 'orig_ip_bytes',
]

df = zscore_normalization(df, cols_to_zscore)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df['type'].values)
df['type'] = le.transform(df['type'])

cols_to_del = [
    'uid',
    'id.orig_h',
    'id.orig_p',
    'id.resp_h',
    'id.resp_p',
    'active.min',
    'service',
    'history',
    'local_orig',
    'local_resp',
    'tunnel_parents',
    'fwd_URG_flag_count',
    'bwd_URG_flag_count',
    'flow_CWR_flag_count',
    'flow_ECE_flag_count',
    ]

df = delete_columns(df,cols_to_del)

print("----------------------")
# Split into input and output variables
x_columns = df.columns.drop('type')
x = df[x_columns].values
y = df['type'].values
print("----------------------")

print(x.shape)
print(y.shape)
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn import tree
# Instantiate model with 1000 decision trees
rf = tree.DecisionTreeClassifier()
# Train the model on training data
rf.fit(x_train, y_train)

predictions = rf.predict(x_test)

from mlciic import functions
from mlciic import metrics as met

met.calculate_metrics("decision tree", y_test, predictions, average='weighted')


#Confusion Matrix 
fig,ax = plt.subplots(figsize=(20, 20))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot(ax=ax)
plt.savefig("confusion_matrix.png")
plt.show()
