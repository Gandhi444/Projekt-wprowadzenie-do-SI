import numpy as np
import cv2


def Detection(Data, Clasifier):
    sift = cv2.SIFT_create()
    flan = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flan)
    vocab = np.load('vocab.npy')
    bow.setVocabulary(vocab)
    counter = 0
    for entry in Data:
        result = []
        image = np.copy(entry['image'])
        size = (image.shape[1], image.shape[0])
       # print(size)
        windowSizes = []
        for i in [5]:
            windowSizes.append((size[0] * i / 10, size[1] * i / 10))
        for window in windowSizes:
            for i in range(np.floor(size[0] / (window[0] / 2)).astype(int)-1):
                for j in range(np.floor(size[1] / (window[1] / 2)).astype(int)-1):
                    image2 = np.copy(image)
                    bounds = np.floor((i * window[0] / 2, i * window[0] / 2 + window[0], j * window[1] / 2,
                                       j * window[1] / 2 + window[1])).astype(int)
                    # print(bounds)
                    # image2 = image2[bounds[2]:bounds[3], bounds[0]:bounds[1]]
                    # cv2.imshow('test', image2)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    kpts = sift.detect(image2, None)
                    desc = bow.compute(image2, kpts)
                    if desc is None:
                        desc = np.zeros((1, 128))
                    prediction = Clasifier.predict(desc)
                    if prediction == 'speedlimit':
                        counter += 1
                        result.append(bounds)
                    # if prediction == 'speedlimit' and entry['label']!='speedlimit':
                    #     cv2.imshow('test',image2)
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
        if len(result) > 0:
            entry['predicted'] = 'speedlimit'
            area = []
            for rectangle in result:
                area.append((rectangle[1] - rectangle[0]) * (rectangle[3] - rectangle[2]))
            entry['bounds'] = result[area.index(min(area))]
        else:
            entry['predicted'] = 'other'
    print(counter)
