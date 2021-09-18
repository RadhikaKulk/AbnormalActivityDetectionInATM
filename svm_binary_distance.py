import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

data = pd.read_csv('features_distance_binary.csv')
X = data.values[:,:2399]
y = data.values[:,2400]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for linear:", metrics.accuracy_score(y_test, y_pred)) 

svclassifier = SVC(kernel = 'poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for polynomial:", metrics.accuracy_score(y_test, y_pred))

svclassifier = SVC(kernel = 'rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for rbf:", metrics.accuracy_score(y_test, y_pred))
