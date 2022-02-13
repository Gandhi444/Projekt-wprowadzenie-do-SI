import numpy as np
import cv2
import os


def Detection(Data, Clasifier):
    sift = cv2.SIFT_create()
    flan = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flan)
    vocab = np.load('vocab.npy')
    bow.setVocabulary(vocab)
    counter = 0
    window_step_coef = 3
    for entry in Data:
        result = []
        image = np.copy(entry['image'])
        size = (image.shape[1], image.shape[0])
        windowSizes = []
        for i in [3, 4,6]:
            windowSizes.append((size[0] * i / 10, size[1] * i / 10))
        for window in windowSizes:
            for i in range(int(size[0] / (window[0] / window_step_coef)) - window_step_coef + 1):
                for j in range(int(size[1] / (window[1] / window_step_coef)) - window_step_coef + 1):
                    image2 = np.copy(image)
                    bounds = np.floor((i * window[0] / window_step_coef, i * window[0] / window_step_coef + window[0],
                                       j * window[1] / window_step_coef,
                                       j * window[1] / window_step_coef + window[1])).astype(int)
                    image2 = image2[bounds[2]:bounds[3], bounds[0]:bounds[1]]
                    kpts = sift.detect(image2, None)
                    desc = bow.compute(image2, kpts)
                    if desc is None:
                        desc = np.zeros((1, 128))
                    prediction = Clasifier.predict(desc)
                    if prediction == 'speedlimit':
                        counter += 1
                        result.append(bounds)
        if len(result) > 0:
            entry['predicted'] = 'speedlimit'
            area = []
            for rectangle in result:
                area.append((rectangle[1] - rectangle[0]) * (rectangle[3] - rectangle[2]))
            entry['bounds'] = result[area.index(min(area))]
        else:
            entry['predicted'] = 'other'


def Classify(Data, Classfier):
    sift = cv2.SIFT_create()
    flan = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flan)
    vocab = np.load('vocab.npy')
    bow.setVocabulary(vocab)
    curentDir = os.getcwd()
    parentDir = os.path.abspath(os.path.join(curentDir, os.pardir))
    for entry in Data:
        path = parentDir + '/test/images/' + entry['file']
        image = cv2.imread(path)
        for Bound in entry['bounds']:
            copiedimg = np.copy(image)
            croped_img = copiedimg[Bound[2]:Bound[3], Bound[0]:Bound[1]]
            kpts = sift.detect(croped_img, None)
            desc = bow.compute(croped_img, kpts)
            if desc is None:
                desc = np.zeros((1, 128))
            prediction=Classfier.predict(desc)[0]
            print(prediction)
