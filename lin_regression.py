import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = self.__prepend_unit_column(X)
        A = X.T.dot(X)
        b = X.T.dot(y)
        self.coef_ = np.linalg.inv(A).dot(b)

    def predict(self, X):
        X = self.__prepend_unit_column(X)
        return np.dot(X, self.coef_)

    @staticmethod
    def __prepend_unit_column(X):
        n, _ = X.shape
        unit_column = np.ones(shape=(n, 1))
        return np.append(unit_column, X, axis=1)
