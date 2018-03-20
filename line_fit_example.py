import random
import matplotlib.pyplot as plt
import numpy as np

from lin_regressor import LinRegressor


def f(x):
    return 2 + 1.5 * x


n = 5
r = range(1, n + 1)
X = list(r)
y = [f(x) + random.uniform(-1, 1) for x in r]

X = np.array([X]).transpose()
y = np.array([y]).transpose()

linReg = LinRegressor()

linReg.fit(X, y)
print(linReg.weights)

r_ = range(0, n + 2)
X_ = np.array([r_]).transpose()
y_ = linReg.predict(X_)

plt.plot(X_, y_, linewidth=2.0)
plt.plot(X, y, 'bo')
plt.axis([0, n + 1, 0, f(n + 1) + 1])
plt.show()
