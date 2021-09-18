import json
import os
from os import path
from pprint import pprint
import math
import csv
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0)
#File names
filenames = ["S1_A9_001"]
#filenames = ["S1_A1_001","S1_A1_002"]

w1, h1 = 6, 550;
#stores distance for each frame
dist = [[0 for x in range(w1)] for y in range(h1)]
#iterate in all video folders
for filehead in filenames:
    itr=0;
    while itr<=550:
        print("In iteration ",itr)
        if itr<10:
            str1 = "C:\\Users\\MITALI\\Desktop\\op\\otput_skpoints\\"+filehead+"\\"+filehead+"_00000000000"+str(itr)+"_keypoints.json"
        elif itr<100:
            str1 = "C:\\Users\\MITALI\\Desktop\\op\\otput_skpoints\\"+filehead+"\\"+filehead+"_0000000000"+str(itr)+"_keypoints.json"
        else:
            str1 = "C:\\Users\\MITALI\\Desktop\\op\\otput_skpoints\\"+filehead+"\\"+filehead+"_000000000"+str(itr)+"_keypoints.json"
        if path.exists(str1):
            data_json = open(str1).read()
            data = json.loads(data_json)
        else:
            break
        w, h = 2, 25;
        arr = [[0 for x in range(w)] for y in range(h)] 
        #arr[25][2] to store 25 X 2 coordinates of 25 joints 
        i = 0
        if not data["people"]:
            print("List is empty")
        else:
            while (i<25):
                arr[i][0] = data["people"][0]["pose_keypoints_2d"][i*3]
                arr[i][1] = data["people"][0]["pose_keypoints_2d"][i*3 + 1]
                i=i+1

        pprint(arr)
        
        base=distance(arr[8][0],arr[8][1],arr[1][0],arr[1][1])
        if base == 0:
            base = 1.0
        dist[itr][0]=distance(arr[4][0],arr[4][1],arr[8][0],arr[8][1])/base
        dist[itr][1]=distance(arr[3][0],arr[3][1],arr[8][0],arr[8][1])/base
        dist[itr][2]=distance(arr[2][0],arr[2][1],arr[8][0],arr[8][1])/base
        dist[itr][3]=distance(arr[5][0],arr[5][1],arr[8][0],arr[8][1])/base
        dist[itr][4]=distance(arr[6][0],arr[6][1],arr[8][0],arr[8][1])/base
        dist[itr][5]=distance(arr[7][0],arr[7][1],arr[8][0],arr[8][1])/base
        itr=itr+1
    pprint("distance\n")
    pprint(dist)
    opfile = 'C:\\Users\\MITALI\\Desktop\\op\\final\\'+filehead+'distFeatures.csv'
    with open(opfile, 'a') as filehandle:
        i=0
        while i < itr:
            j=0
            while j < 6:
                if j<5:
                    filehandle.write('%s,' % dist[i][j])
                else:
                    filehandle.write('%s' % dist[i][j])
                j+=1
            filehandle.write('\n')
            i+=1

             
for filehead in filenames:
    print("HI")
    str = 'C:\\Users\\MITALI\\Desktop\\op\\final\\'+filehead+'distFeatures.csv'
    print("Read from file")
    X=pd.read_csv(str)
    print(X)
    n=10
    # K-Means on the data
    kmeans = KMeans(n_clusters=n,init ='k-means++' ,  random_state=25).fit(X)
    
    print(len(kmeans.labels_))
    my_list = list()
    # take integer value for arr_size
    arr_size = len(kmeans.labels_)//n
    i=0
    while i<n:
        np.set_printoptions(suppress=True)
        j=0
        #store class for each frame of sub-group
        arr= [0] * (arr_size+1)
        # count for each class
        while j<arr_size:
            temp =kmeans.labels_[(i*arr_size)+j]
            print(temp)
            arr[temp]+=1
            j+=1
        maxval=0
        j=0
        label=0
        print('now finding max')
        # Find max (which class is max)
        while j<arr_size:
            if arr[j]>maxval:
                maxval = arr[j]
                label=j
            j+=1
        k=0
        #Append it's centroid 
        print('Appending')
        while k<6:
            my_list.append(kmeans.cluster_centers_[label][k])
            k+=1
        i+=1
    print(kmeans.cluster_centers_)
    print(my_list)
    print(len(my_list))
    with open('kmeansDist_n_10.csv', 'a') as filehandle:
        for listitem in my_list:
            filehandle.write('%s,' % listitem)
        filehandle.write('1\n')
        filehandle.write('\n')
        filehandle.close()

svclassifier_linear = pickle.load(open('svm_oneclass_distance_linear.sav', 'rb'))
svclassifier_rbf = pickle.load(open('svm_oneclass_distance_rbf.sav', 'rb'))
svclassifier_poly = pickle.load(open('svm_oneclass_distance_poly.sav', 'rb'))

min_max_scaler = preprocessing.MinMaxScaler()
data_training = pd.read_csv("kmeansDist_n_10.csv", names=['A','B','C','D','E','F',
                                                            'G','H','I','J','K','L',
                                                            'M','N','O','P','Q','R',
                                                            'S','T','U','V','W','X',
                                                            'Y','Z','AA','AB','AC','AD',
                                                            'AE','AF','AG','AH','AI','AJ',
                                                            'AK','AL','AM','AN','AO','AP',
                                                            'AQ','AR','AS','AT','AU','AV',
                                                            'AW','AX','AY','AZ','BA','BB',
                                                            'BC','BD','BE','BF','BG','BH'])
print(data_training)
print("read file")
X_train = data_training.values[:,:59]
X_train = min_max_scaler.fit_transform(X_train)

x = X_train[0, :59]
#y = y_train[0]
x = x.reshape(1, -1)

print("Linear kernel: ")
y_p = svclassifier_linear.predict(x)
#print(y)
print(y_p)

print("RBF kernel: ")
y_p = svclassifier_rbf.predict(x)
#print(y)
print(y_p)

print("Poly kernel: ")
y_p = svclassifier_poly.predict(x)
#print(y)
print(y_p)


            
        
    
