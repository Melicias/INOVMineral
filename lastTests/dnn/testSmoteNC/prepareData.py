#!pip install pytorch-tabnet wget
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import tree
from xgboost import plot_tree
import pandas as pd
import numpy as np
import os
import wget
from pathlib import Path
import shutil
import gzip
import joblib
from xgboost import XGBClassifier
from mlciic import functions
from mlciic import metrics as met
from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

random_state=42
np.random.seed(random_state)


df_train = pd.read_csv('../../../data/EdgeIIot_train.csv', low_memory=False)
df_test = pd.read_csv('../../../data/EdgeIIot_test.csv', low_memory=False)

#for the SMOTE part, so it can fit in 16gb of RAM
df_before = df_train
df_attacks = df_train[df_train["Attack_type"] != "Normal"]

df_normal = df_train[df_train["Attack_type"] == "Normal"]
df_normal = shuffle(df_normal)
df_normal = df_normal[:100000]
df_train = pd.concat([df_attacks,df_normal])
df_train = shuffle(df_train)

functions.display_information_dataframe(df_train,showCategoricals = True, showDetailsOnCategorical = True, showFullDetails = True)

#print(df_train["Attack_type"].value_counts())

df_train.drop(["Attack_label"], axis=1, inplace=True)
df_test.drop(["Attack_label"], axis=1, inplace=True)

features = [ col for col in df_train.columns if col not in ["Attack_label"]+["Attack_type"]] 

featuresFromStart = [ col for col in df_train.columns if col not in ["Attack_type"]]
categorical_columns = []
for col in df_train.columns[df_train.dtypes == object]:
    if col != "Attack_type":
        categorical_columns.append(col)
        
catIndexs = []
for cc in categorical_columns:
    catIndexs.append(featuresFromStart.index(cc))

df_train = functions.apply_smotenc_bigdata(df= df_train, label= "Attack_type", categorical_indices= catIndexs, random_state= random_state)

#join the 2 df with keys so we can split it
df = pd.concat([df_train,df_test],keys=[0,1])

colunas_one_hot = {}
for coluna in categorical_columns:
    codes, uniques = pd.factorize(df[coluna].unique())
    colunas_one_hot[coluna] = {"uniques": uniques, "codes":codes}
    df[coluna] = df[coluna].replace(colunas_one_hot[coluna]["uniques"], colunas_one_hot[coluna]["codes"])
    
df = pd.get_dummies(data=df, columns=categorical_columns)

df_train,df_test = df.xs(0),df.xs(1)

features = [ col for col in df_train.columns if col not in ["Attack_label"]+["Attack_type"]] 

#Encoding
le = LabelEncoder()
le.fit(df_train["Attack_type"].values)

n_total = len(df_train)
test_indices, valid_indices = train_test_split(range(n_total), test_size=0.2, random_state=random_state)

X_train = df_train[features].values[test_indices]
y_train = df_train["Attack_type"].values[test_indices]
y_train = le.transform(y_train)

X_valid = df_train[features].values[valid_indices]
y_valid = df_train["Attack_type"].values[valid_indices]
y_valid = le.transform(y_valid)

X_test = df_test[features].values
y_test = df_test["Attack_type"].values
y_test = le.transform(y_test)

standScaler = StandardScaler()
model_norm = standScaler.fit(X_train)

X_train = model_norm.transform(X_train)
X_test = model_norm.transform(X_test)
X_valid = model_norm.transform(X_valid)


#FAZER SMOTE AQUI
sm = SMOTE(random_state=random_state,n_jobs=-1)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("-----------lllll")
print(len(le.classes_))

start_time = functions.start_measures()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Define the model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(le.classes_), activation='softmax')) 

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.3,mode="min",patience=10,verbose=1,min_lr=1e-8)
checkpoint = ModelCheckpoint('best_model_multiclass.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=512, callbacks=[monitor, checkpoint])

# Load the best saved model
best_model = load_model('best_model_multiclass.h5')

# Evaluate the best saved model
score = best_model.evaluate(X_test, y_test)
print('')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()


fig, ax = plt.subplots(figsize=(20, 16))

ax.plot(history.history['loss'], label='train')
ax.plot(history.history['val_loss'], label='test')
ax.set_title('Model Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['Train', 'Test'], loc='upper right')
plt.savefig("val_loss.png")
plt.show()


predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

functions.stop_measures(start_time)

met.calculate_metrics("decision tree", y_test, predictions, average='weighted')


#Confusion Matrix 
original_labels_list = le.classes_
fig,ax = plt.subplots(figsize=(20, 20))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = original_labels_list)
cm_display.plot(ax=ax)
plt.savefig("confusion_matrix.png")
plt.show()