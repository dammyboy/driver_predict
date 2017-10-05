from __future__ import division
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/train.csv')
df = df.drop('id', axis = 1)
print Counter(df['target'])

Xdf = df.drop('target', axis = 1)
y = df['target'].values
        
features = Xdf.columns.values

for f in features:

    if 'cat' == f[-3:]:
        Xdf = pd.get_dummies(Xdf, columns = [f])

# print len(features), len(Xdf.columns)
#Bootstrap training
k = 10
while k:
    k-=1
    
    Xdf = Xdf.sample(frac = 1)
    X_train, X_test, y_train, y_test = train_test_split(Xdf.values, y.values, test_size = 0.4)
    sampler = SMOTE()
    X_sampled, y_sampled = sampler.fit_sample(X_train, y_train)

    #Train model
    mod = RandomForestClassifier(max_depth=10, n_estimators=50)
    mod.fit(X_sampled,y_sampled)
    
    #Test model
    y_pred = mod.predict(X_test)
    print classification_report(y_test,y_pred)
