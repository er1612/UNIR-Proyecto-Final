#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import pydotplus


# In[2]:


df = pd.read_csv("diabetes_dataset.csv", sep=',')
# Eliminando valoes nulos
df = df.dropna()
print(df.head())
df.count()


# In[3]:


# Eliminando registros que podrían ser conflictivos
df = df[df.smoking_history != 'No Info']
df = df[df.gender != 'Other']
# Convirtiendo variable categórica gender
df['gender'] = df['gender'].replace(to_replace='Female', value=1)
df['gender'] = df['gender'].replace(to_replace='Male', value=0)
# Convirtiendo variable categórica smoking_history
df['smoking_history'] = df['smoking_history'].replace(to_replace=['never', 'not current'], value=0)
df['smoking_history'] = df['smoking_history'].replace(to_replace=['former', 'current', 'ever'], value=1)
print(df.head())
df.count()


# In[4]:


bosque = RandomForestClassifier(
    n_estimators=100, # Cuántos árboles se van a generar
    criterion="gini", # Criterio de creación
    max_features="sqrt", # Cuántas características se van a tomar en cuenta cada vez que se genere una rama
    bootstrap=True, # Muestreo aleatorio de los datos
    max_samples=2/3, # Qué porcentaje vamos a muestrear
    oob_score=True # Out of the bag son las instancias que fueron excluidas para el sampleo
)

x = df.drop(columns=['diabetes'])
y = df['diabetes']

bosque.fit(x, y)

print(bosque.predict([[1,20,0,0,0,27.32,6.6,85]]))
print(bosque.score(x, y))
print(bosque.oob_score_)


# In[5]:


for arbol in bosque.estimators_:
    tree.plot_tree(
        arbol, 
        feature_names=[
            'gender', 'age', 'hyp', 'heart_d', 'smoking', 'bmi', 'hb1ac', 'glucose'
        ]
    )
    plt.show()

