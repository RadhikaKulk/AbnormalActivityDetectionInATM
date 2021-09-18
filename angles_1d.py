import json
import os
from os import path
from pprint import pprint
import math

def slope(x1, y1, x2, y2):
    if(x2==x1):
        return 1000
    return (float)(y2-y1)/(x2-x1)
angles = [0 for i in range(1600)] 
itr=0;
while itr<=399:
    print("In iteration ",itr)
    if itr<10:
        str1 = "otput_skpoints\\S3_A9_003\\S3_A9_003_00000000000"+str(itr)+"_keypoints.json"
    elif itr<100:
        str1 = "otput_skpoints\\S3_A9_003\\S3_A9_003_0000000000"+str(itr)+"_keypoints.json"
    else:
        str1 = "otput_skpoints\\S3_A9_003\\S3_A9_003_000000000"+str(itr)+"_keypoints.json"
    if path.exists(str1):
        data_json = open(str1).read()
        data = json.loads(data_json)
    

    w, h = 2, 25;
    arr = [[0 for x in range(w)] for y in range(h)] 
    #arr[25][2]
    i = 0
    if not data["people"]:
        print("List is empty")
    else:
        while (i<25):
            arr[i][0] = data["people"][0]["pose_keypoints_2d"][i*3]
            arr[i][1] = data["people"][0]["pose_keypoints_2d"][i*3 + 1]
            i=i+1

    pprint(arr)
    
    
    #angles[1][40]
    
    m1=slope(arr[3][0],arr[3][1],arr[4][0],arr[4][1])
    m2=slope(arr[3][0],arr[3][1],arr[2][0],arr[2][1])
    angles[(itr*4)]=math.degrees(math.atan(abs(m1-m2)/(1+m1*m2)))
    print(angles[(itr*4)])
    m1=slope(arr[3][0],arr[3][1],arr[2][0],arr[2][1])
    m2=slope(arr[2][0],arr[2][1],arr[9][0],arr[9][1])
    angles[(itr*4)+1]=math.degrees(math.atan(abs(m1-m2)/(1+m1*m2)))
    print(angles[(itr*4)+1])
    m1=slope(arr[12][0],arr[12][1],arr[5][0],arr[5][1])
    m2=slope(arr[5][0],arr[5][1],arr[6][0],arr[6][1])
    angles[(itr*4)+2]=math.degrees(math.atan(abs(m1-m2)/(1+m1*m2)))
    print(angles[(itr*4)+2])
    m1=slope(arr[5][0],arr[5][1],arr[6][0],arr[6][1])
    m2=slope(arr[6][0],arr[6][1],arr[7][0],arr[7][1])
    angles[(itr*4)+3]=math.degrees(math.atan(abs(m1-m2)/(1+m1*m2)))
    print(angles[(itr*4)+3])
    itr=itr+1

print("The angles matrix is:\n")
print(angles)
with open('C:\\Users\\MITALI\\Desktop\\features.txt', 'a') as filehandle:  
    for listitem in angles:
        filehandle.write('%s,' % listitem)
    filehandle.write('0\n')
