import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

#read data
data = pd.read_csv("features_binary.csv")
#there are 400 * 4 columns. One more is added for class labels
X = data.values[:,:1599]
y = data.values[:,1600]

#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

#performs min-max normalization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

#linear kernel
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for linear:", metrics.accuracy_score(y_test, y_pred)) 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
svclassifier = SVC(kernel = 'poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for polynomial:", metrics.accuracy_score(y_test, y_pred))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
svclassifier = SVC(kernel = 'rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for rbf:", metrics.accuracy_score(y_test, y_pred))

"""
x = X_train[0, :1599]
y = y_train[0]
x = x.reshape(1, -1)
y_p = svclassifier.predict(x)
print(y)
print(y_p)
"""