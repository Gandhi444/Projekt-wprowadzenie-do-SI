import os
import xml.etree.ElementTree as ET
import shutil
import random


def clear_train_test():
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    folders = [parentDir + '/train/annotations', parentDir + '/train/images', parentDir + '/test/images',
               parentDir + '/test/annotations']
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.xml') or filename.endswith('.png'):
                file_path = os.path.join(folder, filename)
                os.unlink(file_path)


def GenerateDataSets():
    clear_train_test()
    random.seed()
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    total_files = 0
    SpeedLimitIndexs = []
    OtherIndexes = []
    for base, dirs, files in os.walk(parentDir + '/archive/annotations'):
        for Files in files:
            total_files += 1
    for i in range(total_files):
        file_name = 'road' + str(i)
        tree = ET.parse(parentDir + '/archive/annotations/' + file_name + '.xml')
        root = tree.getroot()
        object_type = root[4][0].text
        if object_type == 'speedlimit':
            SpeedLimitIndexs.append(i)
        else:
            OtherIndexes.append(i)
    Counter = 0
    for i in range(len(SpeedLimitIndexs)):
        randomElement = random.randrange(len(SpeedLimitIndexs))
        file_name = 'road' + str(SpeedLimitIndexs[randomElement])
        if Counter == 0:
            shutil.copy(parentDir + '/archive/annotations/' + file_name + '.xml', parentDir + '/test/annotations')
            shutil.copy(parentDir + '/archive/images/' + file_name + '.png', parentDir + '/test/images')
        else:
            shutil.copy(parentDir + '/archive/annotations/' + file_name + '.xml', parentDir + '/train/annotations')
            shutil.copy(parentDir + '/archive/images/' + file_name + '.png', parentDir + '/train/images')
        SpeedLimitIndexs.pop(randomElement)
        Counter = (Counter + 1) % 4

    Counter = 0
    for i in range(len(OtherIndexes)):
        randomElement = random.randrange(len(OtherIndexes))
        file_name = 'road' + str(OtherIndexes[randomElement])
        if Counter == 0:
            shutil.copy(parentDir + '/archive/annotations/' + file_name + '.xml', parentDir + '/test/annotations')
            shutil.copy(parentDir + '/archive/images/' + file_name + '.png', parentDir + '/test/images')
        else:
            shutil.copy(parentDir + '/archive/annotations/' + file_name + '.xml', parentDir + '/train/annotations')
            shutil.copy(parentDir + '/archive/images/' + file_name + '.png', parentDir + '/train/images')
        OtherIndexes.pop(randomElement)
        Counter = (Counter + 1) % 4
