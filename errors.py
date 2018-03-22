import numpy as np


def relative_error(y, y_):
    return np.divide(np.abs(y - y_), y)


def mean_relative_error(y, y_):
    return np.mean(relative_error(y, y_))
