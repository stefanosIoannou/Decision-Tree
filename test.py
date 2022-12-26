import unittest
import decision_tree
import numpy as np


class TestEntropy(unittest.TestCase):
    """ This class tests the entropy calculation for a dummy decision tree """

    def setUp(self):
        self.tree = decision_tree.DecisionTree()

    def test_entropy_0(self):
        entropy = self.tree.compute_entropy(np.array([[0] * 11 + [1] * 9]).T)
        self.assertAlmostEqual(entropy, 0.9928, places=4)

    def test_entropy_1(self):
        entropy = self.tree.compute_entropy(np.array([[0] * 10 + [1] * 10]).T)
        self.assertAlmostEqual(entropy, 1.0, places=4)

    def test_entropy_2(self):
        entropy = self.tree.compute_entropy(np.array([[0] * 10]).T)
        self.assertAlmostEqual(entropy, 0.0, places=4)

    def test_entropy_3(self):
        entropy = self.tree.compute_entropy(np.array([[0] * 8 + [1] * 3]).T)
        self.assertAlmostEqual(entropy, 0.8454, places=4)

    def test_information_gain(self):
        data = np.array([[1] * 11 + [0] * 9]).T
        left_split = np.array([[1] * 3 + [0] * 8]).T
        right_split = np.array([[1] * 8 + [0] * 1]).T
        info_gain = self.tree.compute_information_gain(data, left_split, right_split)
        self.assertAlmostEqual(info_gain, 0.3014, places=4)
