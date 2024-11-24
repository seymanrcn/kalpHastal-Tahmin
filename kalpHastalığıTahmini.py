# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:06:32 2024

@author: cinse
"""
# import libraries
import pandas as pd #data science library
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import warnings
warnings.filterwarnings("ignore")

# load datased and EDA
df = pd.read_csv("heart_disease_uci.csv")
df = df.drop(columns = {"id"}) #id sütununu siliyoruz

df.info()

describe = df.describe()

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure()
sns.pairplot(df,vars=numerical_features, hue="num") #sadece numerikleri gösterir
plt.show()

plt.figure()
sns.countplot(x="num", data = df) #num değerlerin dağılımına bakılıyor

# handling missing value 
print(df.isnull().sum())
df = df.drop(columns = ["ca"])
print(df.isnull().sum()) # ca çıkarıldıktan sonra kayıp veri sayısı

df["trestbps"].fillna(df["trestbps"].median(), inplace = True) #tredbps içindeki boş değerler medyan değerle doldurulur
df["chol"].fillna(df["chol"].median(), inplace = True)

df["fbs"].fillna(df["fbs"].mode()[0], inplace = True)

df["restecg"].fillna(df["restecg"].mode()[0], inplace = True)
df["thalch"].fillna(df["thalch"].median(), inplace = True)
df["exang"].fillna(df["exang"].mode()[0], inplace = True)
df["oldpeak"].fillna(df["oldpeak"].median(), inplace = True)
df["slope"].fillna(df["slope"].mode()[0], inplace = True)
df["thal"].fillna(df["thal"].mode()[0], inplace = True)
#kayıp veriler türlerine göre mod ve medyan ile doldurulur.
print(df.isnull().sum())

# train test split standardizasyon kategorik kodlama
 
X = df.drop(["num"], axis = 1) #num sütunu hedef değişken çıkarılır
y = df["num"]  #hedef değişken elde etme

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.25 , random_state=42)

categorical_features = ["sex", "dataset", "cp","restecg","exang" ,"slope","thal"]
numerical_features = ["age","trestbps","chol","fbs","thalch","oldpeak"]

X_train_num = X_train[numerical_features]

X_test_num = X_test[numerical_features]

scaler = StandardScaler()
# Sayısal verilerin her bir özelliğini, ortalamasını 0 ve standart sapmasını 1 yapacak şekilde dönüştürür. 
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)
#kategorik değerleri scaler işlemi encoder ile 

encoder = OneHotEncoder(sparse_output=False, drop="first")
X_train_cat = X_train[categorical_features]
X_test_cat = X_test[categorical_features]

X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

X_train_transformed = np.hstack((X_train_num_scaled,X_train_cat_encoded))
X_test_transformed = np.hstack((X_test_num_scaled,X_test_cat_encoded))

# modeling : random forest ,knn,voting classifier train and test

rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[
    ("rf",rf),
    ("knn",rf)], voting="soft")

#model eğitimi 

voting_clf.fit(X_train_transformed,y_train)

#test verisi ile tahmin yap 
y_pred = voting_clf.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("confusion matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("classification report : ")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))

sns.heatmap(cm,annot = True, fmt="d" , cmap="Blues",cbar=False)
plt.title("confusion matrix")
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.show()
 











# CM