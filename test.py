import unittest
from numpy.testing import *
from AlgorithmKNN import *

class TestAlgorithmKNN(unittest.TestCase):


    def test_predict(self):
        K = 2
        testDataNoLabels = kNN.loadDataFromCSVfile("tests/test1_data_iris.csv")
        testAnswerLabels = kNN.loadDataFromCSVfile("tests/test1_answer_iris.csv")
        learningData = kNN.loadDataFromCSVfile("tests/test1_learning_iris.csv")

        testAlgorith = kNN(K, learningData)

        answerArray = testAlgorith.predict(testDataNoLabels)
        assert_array_equal(testAnswerLabels,answerArray)

    def test_mostFrequentSpecies(self):
        test_data_array = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica', 'setosa', 'setosa']
        answer = kNN.mostFrequentSpecies(test_data_array)
        self.assertEqual('setosa', answer, 'ups')


    def test_calcEuclideanDistance(self):
        self.assertEqual(kNN.calcEuclideanDistance([0, 1], [0, 2]), 1)
        self.assertEqual(kNN.calcEuclideanDistance([0, -2], [0, 2]), 4)
        self.assertEqual(kNN.calcEuclideanDistance([0, 0], [0, 0]), 0)
        self.assertEqual(kNN.calcEuclideanDistance([0, 1, 0], [0, 2, 0]), 1)

    def test_setK(self):
        learningData = kNN.loadDataFromCSVfile("tests/test1_learning_iris.csv")
        testAlgorith = kNN(5, learningData)

        testAlgorith.setK(2)
        self.assertEqual(testAlgorith.getK(), 2)

 #   def test_score(self):


  #  def test_getLearningSet(self):


   # def test_getK(self):


if __name__ == '__main__':
    unittest.main()