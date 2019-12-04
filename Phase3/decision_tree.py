# CART based Decision Tree
import numpy as np


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.limit = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.classes_number = len(set(y))
        self.features_number = X.shape[1]
        self.main_tree = self.building_the_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def splitting_best(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        parent_number = [np.sum(y == c) for c in range(self.classes_number)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in parent_number)
        index_best, best_limit = None, None
        for idx in range(self.features_number):
            limits, classes = zip(*sorted(zip(X[:, idx], y)))
            left_number = [0] * self.classes_number
            right_number = parent_number.copy()
            for i in range(1, m):
                c = classes[i - 1]
                left_number[c] += 1
                right_number[c] -= 1
                left_gini = 1.0 - sum(
                    (left_number[x] / i) ** 2 for x in range(self.classes_number)
                )
                right_gini = 1.0 - sum(
                    (right_number[x] / (m - i)) ** 2 for x in range(self.classes_number)
                )
                gini = (i * left_gini + (m - i) * right_gini) / m
                if limits[i] == limits[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    index_best = idx
                    best_limit = (limits[i] + limits[i - 1]) / 2
        return index_best, best_limit

    def building_the_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.classes_number)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self.splitting_best(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.limit = thr
                node.left = self.building_the_tree(X_left, y_left, depth + 1)
                node.right = self.building_the_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.main_tree
        while node.left:
            if inputs[node.feature_index] < node.limit:
                node = node.left
            else:
                node = node.right
        return node.predicted_class