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

from matplotlib import pyplot as plt
#%matplotlib inline

#Display information about dataframe
def displayInformationDataFrame(df_cop):
    summary_df = pd.DataFrame(columns=['Data Type', 'Column Name', 'Unique Values'])
    # Iterate through the columns of the original dataframe
    for col in df_cop.columns:
        # Get the data type of the column
        dtype = df_cop[col].dtype
        # Get the column name
        col_name = col
        # Get the unique values of the column
        unique_values = df_cop[col].unique()
        # Append a new row to the summary dataframe
        summary_df = summary_df.append({'Data Type': dtype, 'Column Name': col_name, 'Unique Values': unique_values}, ignore_index=True)
    # display the summary_df
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    #return display(summary_df)

def calcula_metricas(nome_modelo, ground_truth, predicao):
  """
    Funcão Auxiliar para calcular e imprimir métricas: Tx de Acerto, F1, 
    Precisão, Sensibilidade e AUC
  """
  acc = accuracy_score(y_true = ground_truth, y_pred = predicao)
  f1 = f1_score(y_true = ground_truth, y_pred = predicao,average='weighted')
  precision = precision_score(y_true = ground_truth, y_pred = predicao,average='weighted')
  recall = recall_score(y_true = ground_truth, y_pred = predicao,average='weighted')
  #auc_sklearn = roc_auc_score(y_true = ground_truth, y_score = predicao, multi_class='ovr')

  print(f"Desempenho {nome_modelo} - Conjunto de Teste")
  print(f' Taxa de Acerto: {np.round(acc*100,2)}%\n Precisão: {np.round(precision*100,2)}%')
  print(f' Sensibilidade: {np.round(recall*100,2)}%\n Medida F1: {np.round(f1*100,2)}%')
  #print(f' Área sob a Curva: {np.round(auc_sklearn*100,2)}%')

#sys.setrecursionlimit(1000000) 

df_train = pd.read_csv('../../../data/EdgeIIot_train_dummies.csv', low_memory=False)
df_test = pd.read_csv('../../../data/EdgeIIot_test_dummies.csv', low_memory=False)
functions.display_information_dataframe(df_train,showCategoricals = True, showDetailsOnCategorical = True, showFullDetails = True)

df_train.drop(["Attack_label"], axis=1, inplace=True)
df_test.drop(["Attack_label"], axis=1, inplace=True)

features = [ col for col in df_train.columns if col not in ["Attack_label"]+["Attack_type"]] 

#for the SMOTE part, so it can fit in 16gb of RAM
df_before = df_train
df_attacks = df_train[df_train["Attack_type"] != "Normal"]

df_normal = df_train[df_train["Attack_type"] == "Normal"]
df_normal = shuffle(df_normal)
df_normal = df_normal[:750000]
df_train = pd.concat([df_attacks,df_normal])
df_train = shuffle(df_train)

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


sm = SMOTE(random_state=random_state,n_jobs=-1)
X_train, y_train = sm.fit_resample(X_train, y_train)

start_time = functions.start_measures()

# Instantiate model with 1000 decision trees
clf = tree.DecisionTreeClassifier()
# Train the model on training data
clf.fit(X_train, y_train)

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

values = clf.feature_importances_
original_labels_list = le.classes_
importances = [(features[i], np.round(values[i],4)) for i in range(len(features))]


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