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


df_train = pd.read_csv('../../../../data/datasetIot23/EdgeIIot_train_smoteNC.csv', low_memory=False)
df_test = pd.read_csv('../../../../data/datasetIot23/EdgeIIot_test.csv', low_memory=False)

functions.display_information_dataframe(df_train,showCategoricals = True, showDetailsOnCategorical = True, showFullDetails = True)

df_train['is_attack'] = df_train['type'].apply(lambda x: "normal" if x == "normal" else "attack")
print(df_train.groupby('is_attack')['is_attack'].count())
df_test['is_attack'] = df_test['type'].apply(lambda x: "normal" if x == "normal" else "attack")
print(df_test.groupby('is_attack')['is_attack'].count())
features = [ col for col in df_train.columns if col not in ["type","is_attack"]]

train_indices, valid_indices = train_test_split(range(len(df_train)), test_size=0.2, random_state=42)

le = LabelEncoder()
le.fit(df_train["is_attack"].values)

X_train = df_train[features].values[train_indices]
y_train = df_train["is_attack"].values[train_indices]
y_train = le.transform(y_train)

X_valid = df_train[features].values[valid_indices]
y_valid = df_train["is_attack"].values[valid_indices]
y_valid = le.transform(y_valid)

X_test = df_test[features].values
y_test = df_test["is_attack"].values
y_test = le.transform(y_test)

start_time = functions.start_measures()

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

#clf.load_model("modelTabNet.zip")

#predictions = np.array(clf.predict(X_test))
predictions = clf.predict(X_test)

functions.stop_measures(start_time)

met.calculate_metrics("tabnet smotenc normal", y_test, predictions, average='binary')


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
print(importances)

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