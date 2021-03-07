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


def sigmoid(linear_function):
    """ Sigmoid Function """
    sgm = 1 / (1 + np.exp(-linear_function))
    #  Return a probability from a linear function
    return sgm


def cost_function(sample_size, y_trn, y_predicted):
    cost = -1 / sample_size \
           * np.sum(y_trn * np.log(y_predicted) \
                    + (1 - y_trn) * np.log(1 - y_predicted))
    return cost


class LogisticRegression():

    def __init__(self, learning_rate=0.1, n_iters=1000):

        self.alpha = learning_rate
        self.iters = n_iters
        self.info = []

    def __repr__(self):
        df = pd.DataFrame.from_dict(self.info)
        pd.set_option('display.max_columns', None)
        df.set_index('Iteration', inplace=True)
        return f'\n ---------- \n Training Model Coefficients - ' \
               + f'verify the minimum cost: \n ----------\n {df}'

    def fit(self, x_trn, y_trn):
        n_samples, n_features = x_trn.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        tempDict = {}

        for iteration in range(self.iters):
            # Approximate y_hat = 1 / (1 + e(-w.X + bias))
            linear_model = np.dot(x_trn, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Calculate the cost - cross-entropy
            cost = cost_function(n_samples, y_trn, y_predicted)

            # Calculate the difference between predicted y and actual y
            residuals = y_predicted - y_trn

            # Gradient Descent: calculate the partial derivatives of the cost
            # function:
            weight_derivative = (1 / n_samples) * np.dot(x_trn.T, residuals)
            bias_derivative = (1 / n_samples) * np.sum(residuals)

            # Update the parameters - We subtract because the
            # derivatives point in direction of steepest ascent
            # and we need the opposite direction.
            # The size of our update is controlled by the learning rate (alpha).
            self.weights -= self.alpha * weight_derivative
            self.bias -= self.alpha * bias_derivative

            if iteration % 100 == 0:
                tempDict['Iteration'] = iteration
                for i in range(len(self.weights)):
                    tempDict['W' + str(i)] = self.weights[i]
                tempDict['Bias'] = self.bias
                tempDict['Cost'] = cost
                self.info.append(tempDict.copy())

    def predict(self, x_tst):
        linear_model = np.dot(x_tst, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        classes = [1 if i > 0.5 else 0 for i in y_predicted]

        return classes


if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    logistic_classifier = LogisticRegression(learning_rate=0.00001,
                                   n_iters=1000)
    logistic_classifier.fit(X_train, y_train)
    predictions = logistic_classifier.predict(X_test)
    print(logistic_classifier)
    print("----------\nLR classification accuracy:",
          accuracy(y_test, predictions))
