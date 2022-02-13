import Data_sets
import Train
import Test
import Detection
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import math
from skimage.transform import hough_ellipse
from matplotlib import pyplot as plt
#Data_sets.GenerateDataSets()
TrainingData=Train.LoadTrainingData()
Train.CreateBowVocab(TrainingData)
Classifier=Train.Train_KNN_Clasifier(TrainingData)
TestingData=Test.LoadTestingData()
Test.Clasify(TestingData,Classifier)
#Detection.Detection(TestingData,Classifier)
correctznak=0
wrongznak=0
correctother=0
wrongother=0
for entry in TestingData:
    if entry['label']==entry['predicted'] and entry['label']=='speedlimit':
        correctznak+=1
    if entry['label']!=entry['predicted'] and entry['label']=='speedlimit':
        wrongznak+=1
    if entry['label']==entry['predicted'] and entry['label']!='speedlimit':
        correctother+=1
    if entry['label']!=entry['predicted'] and entry['label']!='speedlimit':
        wrongother+=1
print('correctznak',correctznak)
print('wrongznak',wrongznak)
print('correctother',correctother)
print('wrongother',wrongother)
print((correctother+correctznak)/(correctother+correctznak+wrongother+wrongznak))














# Test.Clasify(TestingData,Classifier)
# filename = 'road105'
# curentDir = os.getcwd()
# parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
# tree = ET.parse(parentDir + '/archive/annotations/' + filename + '.xml')
# root = tree.getroot()
# pngName = root[1].text
# object_type = root[4][0].text
# image = cv2.imread(parentDir + '/archive/images/' + filename + '.png')
# entry=image
# size = (entry.shape[1], entry.shape[0])
# windowSizes = []
# for i in range(8, 10):
#     windowSizes.append((size[0] * i / 10, size[1] * i / 10))
# for window in windowSizes:
#     entry = image
#     for i in range(np.floor(size[0] / (window[0] / 4)).astype(int)):
#         for j in range(np.floor(size[1] / (window[1] / 4)).astype(int)):
#             bounds = np.floor((i * window[0] / 4, i * window[0] / 4 + window[0], j * window[1] / 4,
#                                j * window[1] / 4 + window[1])).astype(int)
#             entry2 = entry[bounds[2]:bounds[3], bounds[0]:bounds[1]]
#             print(entry2.shape)
#             cv2.imshow('test', entry2)
#             cv2.waitKey(0)
