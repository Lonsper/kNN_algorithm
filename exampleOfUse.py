from AlgorithmKNN import kNN
import numpy

K = 3
learningData = kNN.loadDataFromCSVfile("iris.data.learning.csv")
testAlgorith = kNN(K, numpy.array(learningData))

testDataNoLabels = kNN.loadDataFromCSVfile("iris.dataNoLabels.test.csv")
answerArray = testAlgorith.predict(testDataNoLabels)
print(answerArray)

testLabels = kNN.loadDataFromCSVfile("iris.correctLabels.test.csv")
correctnessFactor = testAlgorith.score(testDataNoLabels, testLabels)
print(correctnessFactor)

