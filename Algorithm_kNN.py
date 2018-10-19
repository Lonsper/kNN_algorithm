import numpy
import scipy.spatial
import pandas

class kNN:
    def __init__(self, k, learningSet):
        self.k = k
        self.learningSet = learningSet

    def predict(self, inputArray):
        for i in self.learningSet:
            for j in inputArray:
                kNN.calcEuclideanDistans(j, i)

    def calcEuclideanDistans(vectorX, vectorY):
        return scipy.spatial.distance.euclidean(vectorX, vectorY)

    #todo: def score(self, inputArray, label):

    def loadDataFromCSVfile(pathFileName):
        with open(pathFileName, 'r') as csvfile:
            data = pandas.read_csv(csvfile)
            numpyData = numpy.array(data)
        return  numpyData

    def getLearningSet(self):
        return self.learningSet

    def getK(self):
        return self.k

    def setK(self, newK):
        self.k = newK


data = kNN.loadDataFromCSVfile("iris.csv")
testAlgorith = kNN(3, data)
#print(kNN.getLearningSet(testAlgorith))
vectorA = [0,0,0,0,0]
vectorB = [1,1,1,1,1]
vectorC = [1,2,3,4,5]
distans = kNN.calcEuclideanDistans(vectorA, vectorB)
print(distans)
distans = kNN.calcEuclideanDistans(vectorB, vectorC)
print(distans)
distans = kNN.calcEuclideanDistans(vectorA, vectorC)
print(distans)
