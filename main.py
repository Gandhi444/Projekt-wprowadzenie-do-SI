import Data_sets
import Train
#Data_sets.GenerateDataSets()
TrainingData=Train.LoadTrainingData()
#Train.CreateBowVocab(TrainingData)
Train.ExtractFeatures(TrainingData)
Classifier=Train.Train_KNN_Clasifier(TrainingData)
