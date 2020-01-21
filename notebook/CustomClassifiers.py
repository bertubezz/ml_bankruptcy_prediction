##############################################################################
### ICS5110: Applied Machine Learning
###
### Custom Classifiers Implementation
### By Albert Bezzina, Daniel Farrugia, Ivan Salomone
###
### January 2019
##############################################################################

import math
import copy
import numpy as np
import pandas as pd

from scipy import stats

# Base class to easily plug into the sklearn ecosystem e.g. when using Pipelines
from sklearn.base import BaseEstimator

##############################################################################
### Logistic Regression

class CustomLogitRegression(BaseEstimator):
    """Logistic regression classifier.

    Parameters
    ----------
        max_epochs : int
            Iterations upper bound.
        alpha : float
            Learning rate.
        min_gain : float
            Minimum loss difference.
        p_threshold : float
            Class boundary.
        fit_bias : bool
            Add a bias/intercept constant.
        class_balance : bool
            Adjust class balance.
    """

    def __init__(self, max_epochs=1000, alpha=0.1, min_gain=0.0001, p_threshold=0.5,
            fit_bias=True, class_balance=True):

        self.max_epochs = max_epochs
        self.alpha = alpha
        self.min_gain = min_gain
        self.n_nogain = 5
        self.p_threshold = p_threshold
        self.fit_bias = fit_bias
        self.class_balance = class_balance
        self.coef_ = None # Weights to be learned

    ####################
    # Internal functions

    def _add_bias(self, X):
        """Add intercepts to matrix X."""
        return np.insert(X, 0, 1, axis=1)

    def _cost(self, y, y_hat):
        """Finds the prediction cost."""
        return ((-y).T @ np.log(y_hat)) - ((1 - y).T @ np.log(1 - y_hat))

    def _sigmoid(self, Z):
        """Maps Z to a value between 0 and 1."""
        return 1 / (1 + np.exp(-Z))

    ##################
    # Public functions

    def fit(self, X, y):
        """Trains model to predict classes y given X."""

        if self.fit_bias:
            X = self._add_bias(X)

        # Initialise weights
        self.coef_ = np.zeros(X.shape[1])

        # Weighted cross entropy
        n_samples = np.float(y.size)
        y_weights = np.ones(y.size)
        if self.class_balance:
            # Find weights inversely proportional to class frequencies
            class_weights = n_samples / (2 * np.bincount(y))
            y_weights[y == 0] = class_weights[0]
            y_weights[y == 1] = class_weights[1]

        n_nogain = 0
        top_loss = np.Inf

        # Optimise using Stochastic Gradient Descent
        for epoch in range(self.max_epochs):

            # Predict class probabilities
            Z =  X @ self.coef_.T
            y_hat = self._sigmoid(Z)

            # Check if the new coefficients reduce the loss
            loss = (self._cost(y, y_hat) * y_weights).mean()
            if loss > (top_loss - self.min_gain):
                # Loss is increasing, we overshot the minimum?
                n_nogain += 1
            else:
                # Loss is decreasing, keep descending...
                n_nogain = 0

            #if epoch > 0 and epoch % 1000 == 0:
            #    print('{} Loss: {} Top: {}'.format(epoch, loss, top_loss))

            if loss < top_loss:
                top_loss = loss

            # Stop if no improvement in loss is registered
            if n_nogain >= self.n_nogain:
                print('Converged early after {} epochs.'.format(epoch))
                return

            # Find the gradient
            delta = np.matmul(X.T, (y_hat - y) * y_weights) / n_samples

            # Adjust the weights
            self.coef_ -= self.alpha * delta

        print('Reached maximum number of epochs without converging early.')

    def predict_proba(self, X):
        """Find probability of belonging to the true/false class."""

        # Sanity check
        if self.coef_ is None:
            raise RuntimeError('Call fit first!')

        # Add a bias constant
        if self.fit_bias:
            X = self._add_bias(X)

        # Find probability of belonging to true class
        Z = X @ self.coef_.T
        p1 = self._sigmoid(Z)

        # Find probability of belonging to false class
        p0 = 1 - p1

        return np.array([p0, p1]).T

    def predict(self, X):
        """Predicts the classes of X."""
        return self.predict_proba(X)[:,1] >= self.p_threshold

### Logistic Regression
##############################################################################

##############################################################################
### Decision Tree

class _LeafNode():
    """Class that represents a leaf in the decision tree"""
    def __init__(self, y):
        self.outcome = y

    def predict(self, X, proba):
        if proba:
            # Calculate class probality
            bc = np.bincount(self.outcome)
            zeros = bc[0]
            ones = bc[1] if len(bc) == 2 else 0
            return np.array([zeros, ones], dtype=np.float) / len(self.outcome)
        else:
            # Calculate the outcome base on the majority vote
            values, counts = np.unique(self.outcome, return_counts=True)
            return values[counts.argmax()]

class _DecisionNode():
    """Class that represents a decision node in the decision tree"""
    def __init__(self, i_feature, threshold, left_branch, right_branch):
        self.i_feature = i_feature
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch

    def predict(self, X, proba):
        """
        Do a recursive search down the tree and make a prediction of
        the data sample by the outcome value of the leaf that we end
        up at.
        """
        # Choose the feature that we will test
        feature_value = X[self.i_feature]

        # Determine if we will follow left or right branch
        branch = self.right_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= self.threshold:
                branch = self.left_branch
        elif feature_value == self.threshold:
            branch = self.left_branch

        # Test subtree
        return branch.predict(X, proba)

class CustomDecisionTree(BaseEstimator):
    """
    A Decision-tree classifier.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """
    def __init__(self, min_samples_split=2, min_impurity=0, max_depth=float("inf")):
        self.root = None  # Root node
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

    ####################
    # Internal functions

    def _predict(self, X, proba):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.root is None:
            raise RuntimeError('call fit first!')

        return np.array([self.root.predict(X[i, :], proba) for i in range(X.shape[0])])
        
    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and
        respective y on the feature of X which (based on impurity) best separates
        the data.
        """

        n_samples, _ = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            
            impurity, i_feature, value, left_X, right_X, left_y, right_y = \
                self._find_best_split(X, y)
            
            if impurity is not None and impurity > self.min_impurity:
                
                # Build left and right branches
                left_branch = self._build_tree(left_X, left_y, current_depth + 1)
                right_branch = self._build_tree(right_X, right_y, current_depth + 1)
                
                return _DecisionNode(i_feature=i_feature, threshold=value,
                    left_branch=left_branch, right_branch=right_branch)

        # We're at leaf
        return _LeafNode(y)

    def _find_best_split(self, X, y):
        """Find best feature and value for a split. Greedy algorithm."""

        def calculate_entropy(p):
            # _, counts = np.unique(y, return_counts=True)
            # entropy = 0.0
            # for prob in counts / float(len(y)):
            #     entropy -= prob * math.log(prob, 2)
            # return entropy
            p = np.bincount(p) / float(p.shape[0])
            return stats.entropy(p)

        def calculate_information_gain(y, left_y, right_y):
            # p = len(left_y) / len(y)
            # return calculate_entropy(y) - p * \
            #     calculate_entropy(left_y) - (1 - p) * \
            #     calculate_entropy(right_y)
            return calculate_entropy(y) \
                - calculate_entropy(left_y) * (float(left_y.shape[0]) / y.shape[0]) \
                - calculate_entropy(right_y) * (float(right_y.shape[0]) / y.shape[0])

        def find_splits(x):
            """Find all possible split values."""
            split_values = set()

            # Get unique values in a sorted order
            x_unique = list(np.unique(x))
            for i in range(1, len(x_unique)):
                # Find a point between two values
                average = (x_unique[i - 1] + x_unique[i]) / 2.0
                split_values.add(average)

            return list(split_values)

        def split_mask(x, value):
            if isinstance(value, int) or isinstance(value, float):
                left_mask = (x >= value)
                right_mask = (x < value)
            else:
                left_mask = (x == value)
                right_mask = (x != value)
            return left_mask, right_mask

        max_gain, max_i_feature, max_value = None, None, None

        _, n_features = np.shape(X)
        for i_feature in range(n_features):
            column = X[:, i_feature]
            split_values = find_splits(column)
            for value in split_values:
                left_mask, right_mask = split_mask(column, value)
                gain = calculate_information_gain(y, y[left_mask], y[right_mask])

                if (max_gain is None) or (gain > max_gain):
                    max_i_feature, max_value, max_gain = i_feature, value, gain
        
        if max_gain is None:
            return None, None, None, None, None, None, None
        
        left_mask, right_mask = split_mask(X[:, max_i_feature], max_value)
        return max_gain, max_i_feature, max_value, \
            X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    ##################
    # Public functions

    def fit(self, X, y):
        """Trains model to predict classes y given X."""
        if isinstance(X, pd.DataFrame):
            X, y = X.values, y.values

        self.root = self._build_tree(X, y)

    def predict_proba(self, X):
        """Find probability of belonging to true/negative class."""
        return self._predict(X, True)

    def predict(self, X):
        """Predicts the classes of X."""
        return self._predict(X, False)    

### Decision Tree
##############################################################################

##############################################################################
### Bagging

class CustomBagging(BaseEstimator):
    """
    A Bagging classifier.

    Parameters:
    -----------
    base_estimator: object (default=None)
        The base estimator.
        If None, then the base estimator is a decision tree.
    n_estimators: int
        The number of base estimators.
    base_n_features: int or float
        The number of features to draw from X to train the base estimator.
        If float, then base_n_features=round(n_features*base_n_features);
        If None, then base_n_features=round(sqrt(n_features)).
    base_n_samples: int or float
        The number of samples to draw from X to train the base estimator.
        If float, then base_n_samples=round(n_samples*base_n_samples);
        If None, then base_n_samples=round(n_samples/n_estimators*1.5).
    class_balance : bool
            Adjust class balance.
    """
    def __init__(self, base_estimator=None, n_estimators=10, base_n_features=None,
        base_n_samples=None, class_balance=True):
        self.n_estimators = n_estimators
        self.base_n_features = base_n_features
        self.base_n_samples = base_n_samples
        self.class_balance = class_balance

        if base_estimator is None:
            base_estimator = CustomDecisionTree()

        self.estimators = [copy.copy(base_estimator) for _ in range(n_estimators)]

    ##################
    # Public functions

    def fit(self, X, y):
        """Trains model to predict classes y given X."""
        if isinstance(X, pd.DataFrame):
            X, y = X.values, y.values

        n_samples, n_features = np.shape(X)

        if isinstance(self.base_n_features, float):
            self.base_n_features = int(n_features * self.base_n_features)
        elif self.base_n_features is None:
            self.base_n_features = int(math.sqrt(n_features))
        if self.base_n_features > n_features:
            self.base_n_features = n_features

        if isinstance(self.base_n_samples, float):
            self.base_n_samples = int(n_samples * self.base_n_samples)
        elif self.base_n_samples is None:
            self.base_n_samples = int(n_samples/self.n_estimators*1.5)
        if self.base_n_samples > n_samples:
            self.base_n_samples = n_samples
        
        p_y = None
        if self.class_balance:
            # Weighted cross entropy
            # Find weights inversely proportional to class frequencies
            cw = 1 / (2 * np.bincount(y).astype(np.float64))
            p_y = np.ones(len(y))
            p_y[y == 0] = cw[0]
            p_y[y == 1] = cw[1]

        for estimator in self.estimators:
            feature_indices = np.random.choice(range(n_features), size=self.base_n_features, replace=False)
            sample_indices = np.random.choice(range(n_samples), size=self.base_n_samples, replace=False, p=p_y)

            # Save the indices of the features for prediction
            estimator.sample_indices = sample_indices
            estimator.feature_indices = feature_indices

            estimator.fit(X[sample_indices][:, feature_indices], y[sample_indices])

        
    def predict(self, X):
        """Predicts the classes of X."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if len(self.estimators) == 0 or self.estimators[0].root is None:
            raise RuntimeError('call fit first!')

        # call predict_proba and pick the best probable class
        predicted_probabilitiy = self.predict_proba(X)
        return np.argmax(predicted_probabilitiy, axis=1)

    def predict_proba(self, X):
        """Find probability of belonging to true/negative class."""
        
        if isinstance(X, pd.DataFrame):
            X = X.values

        if len(self.estimators) == 0 or self.estimators[0].root is None:
            raise RuntimeError('call fit first!')

        # For each estimator make a prediction based on the features that the estimator has trained on
        all_proba = np.zeros((X.shape[0], 2))
        for estimator in self.estimators:
            all_proba += estimator.predict_proba(X[:, estimator.feature_indices])
            
        return all_proba / len(self.estimators)

### Bagging
##############################################################################

##############################################################################
###  T H E   E N D
##############################################################################
