import json
import os
from os import path
from pprint import pprint
import math
import csv

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0)
#File names
filenames = ["S1_A1_001","S1_A1_002","S1_A1_003","S1_A2_001","S1_A2_002","S1_A2_003",
             "S1_A3_001","S1_A3_002","S1_A3_003","S1_A4_001","S1_A4_002","S1_A4_003",
             "S1_A9_001","S1_A9_002","S1_A9_003","S1_A10_001","S1_A10_002","S1_A10_003",
             "S2_A1_001","S2_A1_002","S2_A3_001","S2_A3_002","S2_A3_003","S2_A4_001",
             "S2_A4_002","S2_A4_003","S2_A9_001","S2_A9_002","S2_A9_003","S2_A10_001",
             "S2_A10_002","S2_A10_003","S3_A1_003","S3_A2_002","S3_A2_003","S3_A3_001",
             "S3_A3_002","S3_A3_003","S3_A4_001","S3_A4_002","S3_A4_003","S3_A9_001",
             "S3_A9_002","S3_A9_003"]

w1, h1 = 6, 550;
#stores distance for each frame
dist = [[0 for x in range(w1)] for y in range(h1)]
#iterate in all video folders
for filehead in filenames:
    itr=0;
    while itr<=550:
        print("In iteration ",itr)
        if itr<10:
            str1 = "otput_skpoints\\"+filehead+"\\"+filehead+"_00000000000"+str(itr)+"_keypoints.json"
        elif itr<100:
            str1 = "otput_skpoints\\"+filehead+"\\"+filehead+"_0000000000"+str(itr)+"_keypoints.json"
        else:
            str1 = "otput_skpoints\\"+filehead+"\\"+filehead+"_000000000"+str(itr)+"_keypoints.json"
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
    opfile = 'C:\\Users\\MITALI\\Desktop\\op\\otput_skpoints\\dist_txt\\'+filehead+'distFeatures.txt'
    with open(opfile, 'a') as filehandle:
        i=0
        while i < itr:
            j=0
            while j < 6:
                filehandle.write('%s,' % dist[i][j])
                j+=1
            filehandle.write('\n')
            i+=1
            
        
    
    
