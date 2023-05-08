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

random_state=42
np.random.seed(random_state)


df_train = pd.read_csv('../../../data/EdgeIIot_train.csv', low_memory=False)
df_test = pd.read_csv('../../../data/EdgeIIot_test.csv', low_memory=False)
df_samples = pd.read_csv('../../../data/samples2.csv', low_memory=False)

df_train = pd.concat([df_train,df_samples])
"""
#for the SMOTE part, so it can fit in 16gb of RAM
df_before = df_train
df_attacks = df_train[df_train["Attack_type"] != "Normal"]

df_normal = df_train[df_train["Attack_type"] == "Normal"]
df_normal = shuffle(df_normal)
df_normal = df_normal[:100000]
df_train = pd.concat([df_attacks,df_normal])
df_train = shuffle(df_train)
"""
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

#df_train = functions.apply_smotenc_bigdata(df= df_train, label= "Attack_type", categorical_indices= catIndexs, random_state= random_state)

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

X_train = df_train[features].values
y_train = df_train["Attack_type"].values
y_train = le.transform(y_train)

X_test = df_test[features].values
y_test = df_test["Attack_type"].values
y_test = le.transform(y_test)

standScaler = StandardScaler()
model_norm = standScaler.fit(X_train)

X_train = model_norm.transform(X_train)
X_test = model_norm.transform(X_test)


start_time = functions.start_measures()

# Instantiate model with 1000 decision trees
clf = tree.DecisionTreeClassifier()
# Train the model on training data
clf.fit(X_train, y_train)

joblib.dump(clf, "./modelDecisionTree.joblib")
#loaded_rf = joblib.load("./random_forest.joblib")

predictions = clf.predict(X_test)

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

#importance features
def sortSecond(val):
	return val[1]

values = clf.feature_importances_
original_labels_list = le.classes_
importances = [(features[i], np.round(values[i],4)) for i in range(len(features))]
importances.sort(reverse=True, key=sortSecond)
importances


feature_names = [imp[0] for imp in importances]
importance_vals = [imp[1] for imp in importances]

# Create a horizontal bar chart
fig, ax = plt.subplots()
ax.barh(range(len(importances)), importance_vals)
ax.set_yticks(range(len(importances)))
ax.set_yticklabels(feature_names, fontsize=3) 

# Add axis labels and title
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importances')

plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')