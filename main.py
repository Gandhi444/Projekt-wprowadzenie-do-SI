import Train
import Detection_Classification

# TODO Jakość kodu i raport (3.5/4)
# TODO Raport malo przejrzysty.

# TODO Skuteczność klasyfikacji 0.864 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji (0/2)

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

