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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
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
import pydot

random_state=42
np.random.seed(random_state)

from matplotlib import pyplot as plt

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
    
def fix_data_mixed_types(df, mixed_columns):    
    # Loop over the mixed columns and clean them
    for col_name in mixed_columns:
        dtype_before = df[col_name].dtype
        
        # Convert non-numeric values to NaN and replace with mean
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        mean = df[col_name].mean()
        df[col_name].fillna(mean, inplace=True)
        
        dtype_after = df[col_name].dtype
        print(f"[INFO] Before: {dtype_before} | After: {dtype_after}")
        
    return df


def aplica_SMOTENC(df, label, categorical_indices = []):
    X = df.copy()
    X=X.drop(columns=[label])
    y=df[label].copy()
    print(f"Started SMOTENC; size of df - {df.size} ")
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=random_state)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    df_smote = pd.DataFrame(X_resampled, columns=X.columns)
    df_smote[label]=y_resampled
    print(f"finished SMOTENC; size of df - {df_smote.size}")
    return df_smote


df = pd.read_csv('../../../data/2022_combined.csv', low_memory=False)
drop_columns = ['uid','id.orig_h','id.orig_p','id.resp_h','id.resp_p']

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df = shuffle(df)

featuresFromStart = [ col for col in df.columns if col not in ["type"]]
    
displayInformationDataFrame(df)


def history_to_int(history):
    hist_list = list(history)
    sum = 0
    mapping = {'s': 1, 'S': 2, 'h': 3, 'H': 4, 'a': 5, 'A': 6, 'd': 7, 'D': 8, 'f': 9, 'F': 10, 'r': 11, 'R': 12, 'c': 13, 'C': 14, 'g': 15, 'G': 16, 't': 17, 'T': 18, 'w': 19, 'W': 20, 'i': 21, 'I': 22, 'q': 23, 'Q': 24, '^': 25}
    for char in hist_list:
        sum = sum + mapping.get(char,0)
    return sum

df['flow_duration'] = df['flow_duration'].str.replace(',','.')
df['flow_duration'] = df['flow_duration'].astype(float)


df = fix_data_mixed_types(df, ['duration','resp_bytes','orig_bytes'])
"""

df['duration'] = df['duration'].str.replace('-','0')
df['duration'] = df['duration'].str.replace(',','.')
df['duration'] = df['duration'].astype(float)

df['resp_bytes'] = df['resp_bytes'].str.replace('-','0')
df['resp_bytes'] = df['resp_bytes'].astype(int)

df['orig_bytes'] = df['orig_bytes'].str.replace('-','0')
df['orig_bytes'] = df['orig_bytes'].astype(int)
"""
df['history'] = df.apply(lambda row: history_to_int(row['history']), axis=1)

categorical_columns = []
for col in df.columns[df.dtypes == object]:
    if col != "type":
        categorical_columns.append(col)
        
catIndexs = []
for cc in categorical_columns:
    catIndexs.append(featuresFromStart.index(cc))
print("catIndexes ---------------")
print(catIndexs)
        
print(categorical_columns)
print(df.columns)

df = aplica_SMOTENC(df, "type", categorical_indices = catIndexs)

colunas_one_hot = {}
for coluna in categorical_columns:
    codes, uniques = pd.factorize(df[coluna].unique())
    colunas_one_hot[coluna] = {"uniques": uniques, "codes":codes}
    df[coluna] = df[coluna].replace(colunas_one_hot[coluna]["uniques"], colunas_one_hot[coluna]["codes"])
    print(coluna)
df = pd.get_dummies(data=df, columns=categorical_columns)
displayInformationDataFrame(df)



df = shuffle(df)
n_total = len(df)

features = [ col for col in df.columns if col not in ["type"]] 

le = LabelEncoder()
le.fit(df["type"].values)



train_val_indices, test_indices = train_test_split(range(n_total), test_size=0.2, random_state=random_state)
train_indices, valid_indices = train_test_split(train_val_indices, test_size=0.25, random_state=random_state) # 0.25 x 0.8 = 0.2

X_train = df[features].values[train_indices]
y_train = df["type"].values[train_indices]
y_train = le.transform(y_train)

X_valid = df[features].values[valid_indices]
y_valid = df["type"].values[valid_indices]
y_valid = le.transform(y_valid)

X_test = df[features].values[test_indices]
y_test = df["type"].values[test_indices]
y_test = le.transform(y_test)

standScaler = StandardScaler()
model_norm = standScaler.fit(X_train)

X_train = model_norm.transform(X_train)
X_test = model_norm.transform(X_test)

#sm = SMOTE(random_state=random_state,n_jobs=-1)
#X_train, y_train = sm.fit_resample(X_train, y_train)

clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95, "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)

max_epochs = 1000 if not os.getenv("CI", False) else 2

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=100,
    batch_size=16384, virtual_batch_size=256
)

saved_filename = clf.save_model('modelTabNet')

#loaded_clf = TabNetClassifier()
#loaded_clf.load_model(saved_filename)

y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"FINAL TEST SCORE FOR : {test_acc}")

predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

mask = np.logical_not(np.equal(y_test, predictions))
print(f"Elements wrong classified: {len(X_test[mask])}; Elements correct: {len(X_test) - len(X_test[mask])}")

original_labels_list = le.classes_
fig,ax = plt.subplots(figsize=(20, 20))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels= original_labels_list)
cm_display.plot(ax=ax)
plt.savefig("confusion_matrix.png")
plt.show()

calcula_metricas("tabnet", y_test, predictions)

# Pull out one tree from the forest
#tre = tree.plot_tree(clf)
# Export the image to a dot file
#dot_data = tree.export_graphviz(clf, out_file='treeMulti.dot', feature_names = features, rounded = True, precision = 1)
# Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('treeMulti.dot')
# Write graph to a png file
#graph.write_png('treeMulti.png')

#print(classification_report(y_test, predictions))

print("balanced_accuracy")
print(balanced_accuracy_score(y_test, predictions))