import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("datasets/train.csav").as_matrix()
clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:21000,1:]
train_label = data[0:1000,0]

clf.fit(xtrain,train_label)

#testing dataset
xtest = data[21000:,1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)
count =0;
for i in range(0,21000):
  count += 1 if p[i] == actual_label[i] else 0
 print("Accuracy=", (count/21000)*100)
 
 
 
 
 ACCURACY->>>83.77
