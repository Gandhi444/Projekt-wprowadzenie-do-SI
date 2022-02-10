import os
import xml.etree.ElementTree as ET
from collections import Counter
import shutil
import random


def clear_train_test():
    import os, shutil
    folders = ['train/annotations', 'train/images', 'test/images', 'test/annotations']
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith('.xml') or filename.endswith('.png'):
                file_path = os.path.join(folder, filename)
                os.unlink(file_path)


def CreateDataSets():
    clear_train_test()
    random.seed()
    print('Starting data set generation')
    total_files = 0
    SpeedLimitIndexs = []
    OtherIndexes = []
    for base, dirs, files in os.walk('archive/annotations'):
        for Files in files:
            total_files += 1
    for i in range(total_files):
        file_name = 'road' + str(i)
        tree = ET.parse('archive/annotations/' + file_name + '.xml')
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
            shutil.copy('archive/annotations/' + file_name + '.xml', 'test/annotations')
            shutil.copy('archive/images/' + file_name + '.png', 'test/images')
        else:
            shutil.copy('archive/annotations/' + file_name + '.xml', 'train/annotations')
            shutil.copy('archive/images/' + file_name + '.png', 'train/images')
        SpeedLimitIndexs.pop(randomElement)
        Counter = (Counter + 1) % 4

    Counter = 0
    for i in range(len(OtherIndexes)):
        randomElement = random.randrange(len(OtherIndexes))
        file_name = 'road' + str(OtherIndexes[randomElement])
        if Counter == 0:
            shutil.copy('archive/annotations/' + file_name + '.xml', 'test/annotations')
            shutil.copy('archive/images/' + file_name + '.png', 'test/images')
        else:
            shutil.copy('archive/annotations/' + file_name + '.xml', 'train/annotations')
            shutil.copy('archive/images/' + file_name + '.png', 'train/images')
        OtherIndexes.pop(randomElement)
        Counter = (Counter + 1) % 4
    print('finished data set generation')
