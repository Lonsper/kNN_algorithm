import numpy
import scipy.spatial
import pandas

class kNN:
    def __init__(self, k, learningSet):
        self.k = k
        self.learningSet = learningSet

    def predict(self, inputArray):
        answerArray = []

        for i in inputArray:
            distansAndSpeciesArray = []

            for j in self.learningSet:
                distansAndSpeciesArray.append([ kNN.calcEuclideanDistans(i[:4], j[:4]), j[4]])
                distansAndSpeciesArray.sort()

            speciesArray = []

            for k in range(self.k):
                speciesArray.append(distansAndSpeciesArray[k][1])

            answerForInstance = kNN.mostFrequentSpecies(speciesArray)
            answerArray.append(answerForInstance)

        return numpy.array(answerArray)

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

    def score(self, inputArray, label):
        goodAnswers = 0
        answerArray = kNN.predict(self, inputArray)

        for i in range(len(label)):
            if (answerArray[i] == label[i]):
                goodAnswers += 1

        return goodAnswers/len(label)


    def loadDataFromCSVfile(pathFileName):

        with open(pathFileName, 'r') as csvfile:
            data = pandas.read_csv(csvfile)
            numpyData = numpy.array(data)

        return numpyData

    def getLearningSet(self):
        return self.learningSet

    def getK(self):
        return self.k

    def setK(self, newK):
        self.k = newK



