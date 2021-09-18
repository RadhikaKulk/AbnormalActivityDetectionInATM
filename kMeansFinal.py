from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
filenames = ["S1_A1_001","S1_A1_002","S1_A1_003","S1_A2_001","S1_A2_002","S1_A2_003",
             "S1_A3_001","S1_A3_002","S1_A3_003","S1_A4_001","S1_A4_002","S1_A4_003",
             "S1_A9_001","S1_A9_002","S1_A9_003","S1_A10_001","S1_A10_002","S1_A10_003",
             "S2_A1_001","S2_A1_002","S2_A3_001","S2_A3_002","S2_A3_003","S2_A4_001",
             "S2_A4_002","S2_A4_003","S2_A9_001","S2_A9_002","S2_A9_003","S2_A10_001",
             "S2_A10_002","S2_A10_003","S3_A1_003","S3_A2_002","S3_A2_003","S3_A3_001",
             "S3_A3_002","S3_A3_003","S3_A4_001","S3_A4_002","S3_A4_003","S3_A9_001",
             "S3_A9_002","S3_A9_003"]
             
for filehead in filenames:
    str = "C:\\Users\\MITALI\\Desktop\\op\\otput_skpoints\\dist_csv\\"+filehead+"distFeatures.csv"
    X=pd.read_csv(str)
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
    with open('C:\\Users\\MITALI\\Desktop\\kmeansDist_n_10.txt', 'a') as filehandle:
        for listitem in my_list:
            filehandle.write('%s,' % listitem)
        filehandle.write('1\n')
        filehandle.close()
