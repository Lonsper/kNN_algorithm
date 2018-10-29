from unittest.mock import _patch

import numpy
import scipy.spatial
import pandas

class kNN:


    def __init__(self, k, learningSet):
        self.k = k
        self.learningSet = learningSet

    def predict(self, inputArray):
        answersArray = []
        widthOfTable = numpy.size(self.learningSet, 1) - 1

        for unknowSpecies in inputArray:

            fullList = []
            for singleExample in self.learningSet:
                distance = kNN.calcEuclideanDistance(unknowSpecies[:widthOfTable], singleExample[:widthOfTable])
                speciesForDistance = singleExample[4]

                fullList.append([distance, speciesForDistance])
                fullList.sort()

            speciesArray = []
            for k in range(self.k):
                speciesArray.append(fullList[k][1])

            answerForInstance = kNN.mostFrequentSpecies(speciesArray)
            answersArray.append([answerForInstance])

        return numpy.array(answersArray)

    def mostFrequentSpecies(speciesArray):
        species = ""
        willChangeSpecies = 0

        maxValue = 0
        index = 0
        uniqueSpeciesArray = []
        for i in speciesArray:
            if(kNN.isUniqueInArray(uniqueSpeciesArray, i)):
                uniqueSpeciesArray.append(i)
                uniqueSpeciesArray.append(1)
            else:
                kNN.addOnePointToSpecies(uniqueSpeciesArray, i)

        for iteration in range(len(uniqueSpeciesArray)):
            if (iteration % 2):
                if (maxValue < uniqueSpeciesArray[iteration]):
                    maxValue = uniqueSpeciesArray[iteration]
                    index = iteration - 1

        return uniqueSpeciesArray[index]

    def addOnePointToSpecies(uniqueSpeciesArray, i):
        for iteration in range(len(uniqueSpeciesArray)):
            if(uniqueSpeciesArray[iteration] == i):
                uniqueSpeciesArray[iteration+1]+=1

        return None

    def isUniqueInArray(counterUniqueSpeciesArray, checkSpecies):
        isUnique = True
        for singleSpecies in counterUniqueSpeciesArray:
            if(checkSpecies == singleSpecies):
                isUnique = False

        return isUnique



    def calcEuclideanDistance(vectorX, vectorY):
        return scipy.spatial.distance.euclidean(numpy.array(vectorX), numpy.array(vectorY))

    def score(self, inputArray, label):
        goodAnswers = 0
        answerArray = kNN.predict(self, inputArray)

        for i in range(len(label)):
            if (answerArray[i] == label[i]):
                goodAnswers += 1

        return goodAnswers/len(label)*100


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




