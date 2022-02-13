import Train
import Detection_Classification

TrainingData = Train.LoadTrainingData()
Train.CreateBowVocab(TrainingData)
Classifier = Train.Train_KNN_Clasifier(TrainingData)

files_to_clasify = []
mode = input()
if mode == 'classify':
    n_files = int(input())
    for i in range(n_files):
        filename = input()
        n_cuts = int(input())
        list_of_bounds = []
        for j in range(n_cuts):
            bounds = list(map(int, input().split()))
            list_of_bounds.append(bounds)
        bufor = {'file': filename, 'bounds': list_of_bounds}
        files_to_clasify.append(bufor)
    Detection_Classification.Classify(files_to_clasify, Classifier)
