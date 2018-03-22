import numpy as np


class Errors(object):

    @staticmethod
    def relative_error(y, y_):
        return np.divide(np.abs(y - y_), y)

    @staticmethod
    def mean_relative_error(y, y_):
        return np.mean(Errors.relative_error(y, y_))
