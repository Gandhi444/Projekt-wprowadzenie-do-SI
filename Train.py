import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random


def isRectangleOverlaping(R1, R2):
    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True


def LoadTrainingData():
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    TrainingData = []
    counter = 0
    for filename in os.listdir(parentDir + '/train/annotations'):
        counter = (counter + 1) % 4
        if filename.endswith('.xml'):
            tree = ET.parse(parentDir + '/train/annotations/' + filename)
            root = tree.getroot()
            pngName = root[1].text
            object_type = root[4][0].text
            image = cv2.imread(parentDir + '/train/images/' + pngName)
            boundbox = [root[4][5][0].text, root[4][5][1].text, root[4][5][2].text, root[4][5][3].text]
            for i in range(0, len(boundbox)):
                boundbox[i] = int(boundbox[i])
            image2 = image[boundbox[1]:boundbox[3], boundbox[0]:boundbox[2]]
            if object_type == 'speedlimit':
                TrainingData.append({"label": object_type, "image": image2})
            else:
                TrainingData.append({"label": 'other', "image": image2})
            if counter == 0:
                for i in range(500):
                    w = random.randrange(20, 100)
                    h = random.randrange(20, 100)
                    xstart = random.randrange(100)
                    ystart = random.randrange(100)
                    boundbox2 = [xstart, ystart, xstart + w, ystart + h]
                    if not isRectangleOverlaping(boundbox, boundbox2):
                        break
                image2 = image[boundbox2[1]:boundbox2[3], boundbox2[0]:boundbox2[2]]
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
    sift = cv2.SIFT_create()
    flan = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flan)
    vocab = np.load('vocab.npy')
    bow.setVocabulary(vocab)
    for entry in Data:
        kpts = sift.detect(entry['image'], None)
        desc = bow.compute(entry['image'], kpts)
        if desc is not None:
            entry['descriptor'] = desc
        else:
            entry['descriptor'] = np.zeros((1, 128))


def Train_KNN_Clasifier(Data):
    ExtractFeatures(Data)
    Classifier = KNeighborsClassifier(6)
    x = []
    y = []
    for entry in Data:
        y.append(entry['label'])
        x.append(np.squeeze(entry['descriptor'], 0))
    Classifier.fit(x, y)
    return Classifier
