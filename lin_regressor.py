import numpy as np


class LinRegressor(object):
    weights = None

    def fit(self, X, y):
        X = self.__prepend_unit_column(X)
        n, m = X.shape
        A = np.array([[np.dot(X[:, i], X[:, j]) for j in range(m)] for i in range(m)])
        b = np.array([[np.dot(X[:, i], y[:, 0]) for i in range(m)]]).transpose()
        self.weights = np.linalg.inv(A).dot(b)

    def predict(self, X):
        X = self.__prepend_unit_column(X)
        return np.dot(X, self.weights)

    @staticmethod
    def __prepend_unit_column(X):
        h, _ = X.shape
        unit_column = np.full((h, 1), 1)
        return np.append(unit_column, X, axis=1)
