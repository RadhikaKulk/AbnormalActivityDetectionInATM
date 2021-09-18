import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

data_training = pd.read_csv("features_oneclass_distance_train.csv")
data_testing = pd.read_csv("features_oneclass_distance_test.csv")

X_train = data_training.values[:,:2399]
y_train = data_training.values[:,2400]

X_test = data_testing.values[:,:2399]
y_test = data_testing.values[:,2400]

#normalization
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

svclassifier = svm.OneClassSVM(kernel="linear")
svclassifier.fit(X_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for linear:", metrics.accuracy_score(y_test, y_pred)) 

svclassifier = svm.OneClassSVM(kernel="rbf")
svclassifier.fit(X_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for rbf:", metrics.accuracy_score(y_test, y_pred))

svclassifier = svm.OneClassSVM(kernel="poly")
svclassifier.fit(X_train)
y_pred = svclassifier.predict(X_test)
print("Accuracy for poly:", metrics.accuracy_score(y_test, y_pred)) 
