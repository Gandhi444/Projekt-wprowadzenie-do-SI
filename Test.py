import cv2
import os
import xml.etree.ElementTree as ET
import Train
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def LoadTestingData():
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    TestingData = []
    for filename in os.listdir(parentDir + '/test/annotations'):
        if filename.endswith('.xml'):
            tree = ET.parse(parentDir + '/test/annotations/' + filename)
            root = tree.getroot()
            pngName = root[1].text
            object_type = root[4][0].text
            image = cv2.imread(parentDir + '/test/images/' + pngName)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = cv2.medianBlur(image, 3)
            boundbox = [root[4][5][0].text, root[4][5][1].text, root[4][5][2].text, root[4][5][3].text]
            for i in range(0, len(boundbox)):
                boundbox[i] = int(boundbox[i])
            image = image[boundbox[1]:boundbox[3], boundbox[0]:boundbox[2]]
        if object_type == 'speedlimit':
            TestingData.append({"label": object_type, "image": image})
        else:
            TestingData.append({"label": 'other', "image": image})
    return TestingData

def Clasify(Data,Clasifier):
    Train.ExtractFeatures(Data)
    x=[]
    for entry in Data:
        x.append(np.squeeze(entry['descriptor'], 0))
    predictions=Clasifier.predict(x)
    for i in range(len(predictions)):
        Data[i]['predicted']=predictions[i]
