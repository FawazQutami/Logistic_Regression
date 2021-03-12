# File Name: Logistic_Regression.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def accuracy(y_tst, y_predicted):
    acc = np.sum(y_tst == y_predicted) / len(y_tst)
    # Return accuracy
    return acc


def log_function(sample_size, y_trn, y_predicted):
    """
    Cross-entropy
    :param sample_size: {int}
    :param y_trn: {array_like}
    :param y_predicted: {array_like}
    :return: {float}
    """
    log_summation = np.sum(y_trn * np.log(y_predicted)
                           + (1 - y_trn) * np.log(1 - y_predicted))
    cost = - 1 / sample_size * log_summation

    return cost


def sigmoid(linear_function):
    """
    Sigmoid Function returns a probability from a linear function
    :param linear_function: {array_like}
    :return: {float}
    """
    sgm = 1 / (1 + np.exp(-linear_function))

    return sgm


class LogisticRegression:
    """
    Logistic Regression - a form of Binary Regression
    """

    def __init__(self, alpha=0.1, n_iter=1000):
        """
        Class Constructor
        :param alpha: float (between 0.0 and 1.0)
        :param n_iter: int (epochs over the training set)
        """
        # Initialize the parameters
        self.alpha = alpha
        self.n_iter = n_iter

        # Initialize the attributes
        self.weights = None
        self.bias = None

        self.info = []

    def __repr__(self):
        """
        Class Representation:  represent a class's objects as a string
        :return: string
        """
        df = pd.DataFrame.from_dict(self.info)
        pd.set_option('display.max_columns', None)
        df.set_index('Iteration', inplace=True)
        return f'\n ---------- \n Training Model Coefficients - ' \
               + f'verify the minimum cost: \n ----------\n {df}'

    def fit(self, x_trn, y_trn):
        """
        Fit Method
        :param x_trn: [array-like]
            it is n x m shaped matrix where n is number of samples
                                 and m is number of features
        :param y_trn: [array-like]
            it is n shaped matrix where n is number of samples
        :return: self:object
        """
        # Initialize the parameters
        n_samples, n_features = x_trn.shape
        # Initialize the weights of size n_features with zeros
        self.weights = np.zeros(n_features)
        # Initialize the bias with zero
        self.bias = 0

        temp_dict = {}

        for iteration in range(self.n_iter):
            #  y_hat = w.X + bias
            linear_model = self.linear_function(x_trn)
            # Sigmoid Function 1 / (1 + e(-w.X + bias))
            # returns a probability from a linear function
            y_predicted = sigmoid(linear_model)

            # Calculate the cost - cross-entropy
            cost = log_function(n_samples, y_trn, y_predicted)

            # Calculate the difference between predicted y and actual y
            residuals = y_predicted - y_trn

            """ Gradient Descent -- BackProp """
            self.gradient_descent(n_samples, x_trn, residuals)

            temp_dict['Iteration'] = iteration
            for i in range(len(self.weights)):
                temp_dict['W' + str(i)] = self.weights[i]
            temp_dict['Bias'] = self.bias
            temp_dict['Cost'] = cost
            self.info.append(temp_dict.copy())

    def linear_function(self, x):
        """
        Linear Function (y = w.x + b)
        :param x: {array-like}
        :return: {array-like}
        """
        return np.dot(x, self.weights) + self.bias

    def gradient_descent(self, n_samples, x_trn, residuals):
        """
        Gradient Descent -- BackProp
        :param n_samples: {int}
        :param x_trn: {array_like}
        :param residuals: {array_like}
        :return: None
        """
        # Calculate the partial derivatives of the cost function:
        weight_derivative = (1 / n_samples) * np.dot(x_trn.T, residuals)
        bias_derivative = (1 / n_samples) * np.sum(residuals)

        """ 
        Update the parameters - We subtract because the
        # derivatives point in direction of steepest ascent
        # and we need the opposite direction.
        # The size of our update is controlled by the learning rate (alpha).
        """
        self.weights -= self.alpha * weight_derivative
        self.bias -= self.alpha * bias_derivative

    def predict(self, x_tst):
        """
        Prediction
        :param x_tst:{array_like}
        :return: {array_like} (0's and 1's)
        """
        linear_model = self.linear_function(x_tst)
        # Sigmoid Function returns a probability from a linear function
        sigmoid_function = sigmoid(linear_model)

        y_predicted = [1 if i > 0.5 else 0 for i in sigmoid_function]

        return y_predicted


if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    logistic_classifier = LogisticRegression(alpha=0.00001,
                                             n_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    predictions = logistic_classifier.predict(X_test)
    print(logistic_classifier)
    print("----------\nLR classification accuracy:",
          accuracy(y_test, predictions))
