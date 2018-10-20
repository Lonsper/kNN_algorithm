from AlgorithmKNN import kNN

K = 3
learningData = kNN.loadDataFromCSVfile("iris.data.learning.csv")
testAlgorith = kNN(K, learningData)

testDataNoLabels = kNN.loadDataFromCSVfile("iris.dataNoLabels.test.csv")
answerArray = testAlgorith.predict(testDataNoLabels)
print(answerArray)

testLabels = kNN.loadDataFromCSVfile("iris.correctLabels.test.csv")
correctnessFactor = testAlgorith.score(testDataNoLabels, testLabels)
print(correctnessFactor)