import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def LoadTrainingData():
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    TrainingData = []
    for filename in os.listdir(parentDir + '/train/annotations'):
        if filename.endswith('.xml'):
            tree = ET.parse(parentDir + '/train/annotations/' + filename)
            root = tree.getroot()
            pngName = root[1].text
            object_type = root[4][0].text
            image = cv2.imread(parentDir + '/train/images/' + pngName)
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           # image=cv2.medianBlur(image,3)
            boundbox = [root[4][5][0].text, root[4][5][1].text, root[4][5][2].text, root[4][5][3].text]
            for i in range(0, len(boundbox)):
                boundbox[i] = int(boundbox[i])
            image2 = image[boundbox[1]:boundbox[3], boundbox[0]:boundbox[2]]
            if object_type=='speedlimit':
                #TrainingData.append({"label": object_type, "image": image})
                TrainingData.append({"label": object_type, "image": image2})
            else:
                #TrainingData.append({"label": 'other', "image": image})
                TrainingData.append({"label": 'other', "image": image2})
    return TrainingData


def CreateBowVocab(Data):
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)
    sift = cv2.SIFT_create()
    for entry in Data:
        kpts = sift.detect(entry['image'], None)
        kpts, desc = sift.compute(entry['image'], kpts)
        if desc is not None:
            bow.add(desc)
    vocab = bow.cluster()
    np.save('vocab.npy', vocab)
def ExtractFeatures(Data):
    sift=cv2.SIFT_create()
    flan=cv2.FlannBasedMatcher_create()
    bow=cv2.BOWImgDescriptorExtractor(sift,flan)
    vocab=np.load('vocab.npy')
    bow.setVocabulary(vocab)
    for entry in Data:
        kpts=sift.detect(entry['image'],None)
        desc=bow.compute(entry['image'],kpts)
        if desc is not None:
            entry['descriptor']=desc
        else:
            entry['descriptor']=np.zeros((1,128))
def Train_KNN_Clasifier(Data):
    ExtractFeatures(Data)
    Classifier=KNeighborsClassifier(7)
    x=[]
    y=[]
    for entry in Data:
        y.append(entry['label'])
        x.append(np.squeeze(entry['descriptor'],0))
    Classifier.fit(x,y)
    return Classifier