import numpy
import scipy.spatial
import pandas

class kNN:
    def __init__(self, k, learningSet):
        self.k = k
        self.learningSet = learningSet

    def predict(self, inputArray):

        for i in inputArray:

            distansAndSpeciesArray = []
            for j in self.learningSet:
                distansAndSpeciesArray.append([ kNN.calcEuclideanDistans(i[:4], j[:4]), j[4]])
                distansAndSpeciesArray.sort()

            speciesArray = []
            for k in range(self.k):
                speciesArray.append(distansAndSpeciesArray[k][1])

            answerForExample = kNN.mostFrequentSpecies(speciesArray)
            print("%s = %s" % (speciesArray, answerForExample))

    def mostFrequentSpecies(speciesArray):
        species = ""
        willChangeSpecies = 0
        for i in speciesArray:

            if(species == "" or willChangeSpecies == 0):
                species = i
                willChangeSpecies += 1
            elif(species == i):
                willChangeSpecies += 1
            else:
                willChangeSpecies -= 1

        return species




    def calcEuclideanDistans(vectorX, vectorY):
        return scipy.spatial.distance.euclidean(numpy.array(vectorX), numpy.array(vectorY))

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

K = 3

learningData = kNN.loadDataFromCSVfile("iris.data.learning.csv")
testData = kNN.loadDataFromCSVfile("iris.data.test.csv")
testAlgorithClass = kNN(K, learningData)
kNN.predict(testAlgorithClass, testData)

