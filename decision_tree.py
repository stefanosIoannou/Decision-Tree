from copy import deepcopy
import numpy as np
from metrics import evaluate
from abc import ABC
from typing import Tuple


class Model(ABC):
    def __init__(self):
        self.root = None

    def fit(self, x, y): pass

    def predict(self, x): pass

    def reset_params(self): pass

    def post_order_pruning(self, train, val, root): pass

    def compute_depth(self): pass


class DecisionTree(Model):

    def __init__(self):
        """
        Continuous Decision Tree Classifier.
        """
        self.root = None
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x: np.array, y: np.array):
        """
        Fit the training data to the classifier.

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)
            y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.x = x
        self.y = y

        self.root = self.decision_tree_learning(data=np.c_[x, y], depth=0)

    def predict(self, x: np.array) -> np.array:
        """
        Perform prediction given some data (without labels).

        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
            y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """
        return np.apply_along_axis(self._predict_recursively, 1, x, self.root)

    def _predict_recursively(self, x: np.array, node: dict) -> int:
        """
        Search through the tree for the appropriate label for the data.

        Args:
            x: Instance, numpy array with shape (K,)
            node: dictionary representing a tree node

        Returns:
            (int) a prediction value
        """
        if node['leaf']:
            return node['value']
        if x[node['attribute']] <= node['value']:
            return self._predict_recursively(x, node=node['left'])

        return self._predict_recursively(x, node=node['right'])

    def reset_params(self):
        """
        Reset Decision Tree Model
        """
        self.root = None
        self.x = np.array([])
        self.y = np.array([])

    def decision_tree_learning(self, data: np.array, depth: int) -> dict:
        """ 
        Main decision tree 'building' recursive function that
        follows the pseudocode given in the CW sheet.

        Args:
            data (np.ndarray): Instances + Class Labels, numpy array with shape (N,K)
            depth (int): Depth of the current tree, Integer

        Returns:
            a dictionary representing a tree node
        """
        # If all samples have the same label, this is a leaf node
        unique_labels = np.unique(data[:, -1])
        if len(unique_labels) == 1:
            return {
                'attribute': None,
                'value': unique_labels[0],
                'left': None,
                'right': None,
                'leaf': True
            }
        else:
            split_attribute, split_value = self.find_split_values(data)
            left_data, right_data = self.split_data(data, split_attribute, split_value)

            # Recursive calls to both branches with the split data
            left_branch = self.decision_tree_learning(left_data, depth + 1)
            right_branch = self.decision_tree_learning(right_data, depth + 1)

            # Return new internal node
            return {
                'attribute': split_attribute,
                'value': split_value,
                'left': left_branch,
                'right': right_branch,
                'leaf': False
            }

    def find_split_values(self, data: np.array) -> Tuple[int, float]:
        """
        Return the attribute and the value that results in the highest information gain.

        Args:
            data (np.ndarray): Instances + Class Labels, numpy array with shape (N,K)

        Returns:
            best_attribute (int): The best attribute to split on.
            best_split (float): The best split value.
        """
        # This will record possible split values for all attributes
        possible_splits = {}

        # Iterate through all the columns (attributes) except the last (label)
        for i in range(len(data[1]) - 1):

            # Grab unique i_th column values and sort them in ascending order
            sorted_filtered_data = np.sort(np.unique(data[:, i]))

            # Dict of possible split values for a given attribute i
            possible_splits[i] = []
            for k in range(sorted_filtered_data.shape[0]):
                if k != 0:
                    current = sorted_filtered_data[k]
                    previous = sorted_filtered_data[k - 1]
                    possible_split_value = (current + previous) / 2
                    possible_splits[i].append(possible_split_value)

        return self.find_best_split(data, possible_splits)

    def find_best_split(self, data: np.array, possible_splits: dict) -> Tuple[int, float]:
        """
        Find the most informative attribute by calculating the information gain
        resulting from the choice of attribute and mid-point.
        
        Args:
            data (np.ndarray): Data to be split 
            possible_splits (dict): List of all possible split points

        Returns:
            best_attribute (int): The best attribute to split on.
            best_split (float): The best split value.
        """
        curr_information_gain = -1  # placeholder
        best_attribute = None
        best_split = None

        for attribute_index in possible_splits:
            for mid_point in possible_splits[attribute_index]:
                left, right = self.split_data(data, attribute_index, mid_point)
                information_gain = self.compute_information_gain(data, left, right)
                if information_gain > curr_information_gain:
                    curr_information_gain = information_gain
                    best_attribute = attribute_index
                    best_split = mid_point

        return best_attribute, best_split

    @staticmethod
    def split_data(data: np.array, attribute: int, split_value: float) -> np.array:
        """
        Split the dataset according to the split value for a particular attribute.

        Args:
            data (np.ndarray): Data to be split
            attribute: Attribute that will be split according the value
            split_value: Value that will separate the data

        Returns:
            Left data (np.ndarray) and right data (np.ndarray)
        """
        return data[data[:, attribute] <= split_value, :], data[data[:, attribute] > split_value, :]

    def compute_information_gain(self, data: np.array, left_split: np.array, right_split: np.array) -> float:
        """
        Return information gain for a binary tree according to the formula:
        IG(dataset, subsets) = H(dataset) - (|S_left|/|dataset| * H(S_left) +
        |S_right|/|dataset| * H(S_right).
        
        Args:
            data (np.ndarray): Data before split
            left_split (np.ndarray): Left split data
            right_split (np.ndarray): Right split data

        Returns:
            Information gain (float) [0, 1]
        """
        left_proportion = len(left_split) / len(data)
        right_proportion = len(right_split) / len(data)
        return self.compute_entropy(data) - (left_proportion * self.compute_entropy(left_split)
                                             + right_proportion * self.compute_entropy(right_split))

    @staticmethod
    def compute_entropy(data: np.array):
        """
        Compute and return entropy.
        
        Args:
            data (np.ndarray): Data to compute the entropy

        Returns:
            Entropy (float) [0, 1].
        """
        labels = data[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / labels.size
        entropy = np.sum(-p * np.log2(p))

        return entropy

    @staticmethod
    def set_node_value(node: dict, attribute: int, value: float, left: np.array, right: np.array, leaf: bool):
        """
        Sets the values of a node.

        Args:
            node: Node to change
            attribute: Attribute tag
            value: Value tag
            left: Left node
            right: Right node
            leaf: Boolean to denote if it is a leaf node
        """
        node['attribute'] = attribute
        node['value'] = value
        node['left'] = left
        node['right'] = right
        node['leaf'] = leaf

    def post_order_pruning(self, training_data: np.array, validation_data: np.array, node: dict):
        """
        Pruning performed during postorder traversal of the tree

        Args:
            training_data: Training data
            validation_data: Validation data
            node: current node visited in the traversal
        """
        if node['leaf']:
            return

        attribute, value = node['attribute'], node['value']
        left_data, right_data = self.split_data(training_data, attribute, value)

        self.post_order_pruning(left_data, validation_data, node['left'])
        self.post_order_pruning(right_data, validation_data, node['right'])

        # Here we do pruning and evaluation
        if node['left']['leaf'] and node['right']['leaf']:
            # Evaluate the old tree with accuracy
            old_tree_performance = evaluate(validation_data, self)

            # Take the majority label given the data
            new_label = np.argmax(np.bincount(training_data[:, -1].astype(int)))
            old_node = deepcopy(node)  # copy the node that we have rn

            # Transform this node to the leaf node
            self.set_node_value(node, None, new_label, None, None, True)

            # Evaluate the new tree
            new_tree_performance = evaluate(validation_data, self)

            # If the validation metric is worse go back to the previous node
            if new_tree_performance < old_tree_performance:
                self.set_node_value(node, old_node['attribute'], old_node['value'],
                                    old_node['left'], old_node['right'], old_node['leaf'])

    def compute_depth(self) -> int:
        """
        Compute the depth of the tree

        Returns:
            depth (int) - The depth of the tree
        """

        def compute_depth_recursively(root: dict) -> int:
            """
            Compute the depth of the tree recursively

            Args:
                root (Dict): the dictionary representation of the tree

            Returns:
                depth (int) - The depth of the tree
            """

            if root["leaf"]:
                return 0

            left = root['left']
            right = root['right']

            return 1 + max(compute_depth_recursively(left), compute_depth_recursively(right))

        return compute_depth_recursively(self.root)
