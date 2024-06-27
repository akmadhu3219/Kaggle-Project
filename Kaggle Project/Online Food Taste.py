import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
from colorama import Fore, Style 

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from lightgbm import LGBMClassifier 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_tr = pd.read_csv('onlinefoods.csv')

df_tr.rename(columns={'Output': 'state_of_order'},inplace=True)

df_tr = df_tr.drop('Unnamed: 12',axis=1)

def print_dataset_analysis(train_dataset, n_top=5):
    print(f" Top {n_top} rows of Dataset:")
    print(train_dataset.head(n_top))

    print("\n Summary of Dataset:")
    print(train_dataset.describe())

    print("\n Null Values in Dataset:")
    train_null_count = train_dataset.isnull().sum()
    if train_null_count.sum() == 0:
        print("No null values in the training dataset.")
    else:
        print("Training Dataset:")
        print(train_null_count[train_null_count > 0])

    print("\n Duplicate Values in Dataset:")
    train_duplicates = train_dataset.duplicated().sum()
    print(f"Training Dataset: {train_duplicates} duplicate rows found.")

    print("\n Number of Rows and Columns:")
    print(f"Training Dataset: Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}")

def print_unique_values(train_dataset):
    print("Unique Values in Dataset:")
    unique_values_table = pd.DataFrame({
        'Column Name': train_dataset.columns,
        'Data Type': [train_dataset[col].dtype for col in train_dataset.columns],
        'Unique Values': [train_dataset[col].unique()[:7] for col in train_dataset.columns]
    })
    print(unique_values_table)

print_dataset_analysis(df_tr, n_top=5)
print_unique_values(df_tr)


df_tr['Monthly Income'] = df_tr['Monthly Income'].str.replace('.', '', regex=False)

df_tr['Educational Qualifications'] = df_tr['Educational Qualifications'].str.replace('.', '', regex=False)

df_tr = df_tr.drop_duplicates()

df_tr.shape

background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Marital Status", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
            bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Marital Status in Dataset")
plt.xlabel("Count")
plt.ylabel("Marital Status")
plt.show()

background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Gender", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
            bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Gender in dataset")
plt.xlabel("Count")
plt.ylabel("Gender")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Occupation", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
            bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Occupation in the Dataset")
plt.xlabel("Count")
plt.ylabel("Occupation")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Monthly Income", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
            bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Monthly Income in the Dataset")
plt.xlabel("Count")
plt.ylabel("Monthly Income")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Educational Qualifications", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Educational Qualifications in the dataset")
plt.xlabel("Count")
plt.ylabel("Educational Qualifications")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="state_of_order", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("state_of_order in the dataset")
plt.xlabel("Count")
plt.ylabel("state_of_order")
plt.show()


background_color = '#5fa1bc'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": background_color})
plt.subplots(figsize=(10, 5))
p = sns.countplot(y="Feedback", data=df_tr, palette='magma', edgecolor='white', linewidth=2, width=0.7)
for container in p.containers:
    plt.bar_label(container, label_type="center", color="black", fontsize=17, weight='bold', padding=6, position=(0.5, 0.5),
                bbox={"boxstyle": "round", "pad": 0.2, "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 1})
plt.title("Feedback in the dataset")
plt.xlabel("Count")
plt.ylabel("Feedback")
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="Age", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["Age"].mean()
median_value = df_tr["Age"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Distribution of Age in dataset with Mean and Median")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="Family size", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["Family size"].mean()
median_value = df_tr["Family size"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Distribution of Family size in dataset with Mean and Median")
plt.xlabel("Family Size")
plt.ylabel("Count")
plt.legend()
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="latitude", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["latitude"].mean()
median_value = df_tr["latitude"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Distribution of latitude in dataset with Mean and Median")
plt.xlabel("latitude")
plt.ylabel("Count")
plt.legend()
plt.show()


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#5fa1bc"})
cmap = sns.color_palette("magma", as_cmap=True)
plt.figure(figsize=(10, 6))
histplot = sns.histplot(data=df_tr, x="longitude", bins=20, palette=cmap, edgecolor='white', kde=True)
histplot.get_lines()[0].set_color("#4cc9f0")
mean_value = df_tr["longitude"].mean()
median_value = df_tr["longitude"].median()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.title("Distribution of longitude in dataset with Mean and Median")
plt.xlabel("longitude")
plt.ylabel("Count")
plt.legend()
plt.show()

fig = px.scatter_geo(
    df_tr, 
    lat='latitude', 
    lon='longitude', 
    text='Pin code', 
    title='Geographical Chart with Pincodes'
)
fig.show()

# Splitting figures and targets
X = df_tr.drop('Feedback', axis=1)
y = df_tr['Feedback']

#Column Transformer
numeric_features = ['Age', 'Family size', 'Pin code', 'longitude', 'latitude']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'state_of_order']
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")