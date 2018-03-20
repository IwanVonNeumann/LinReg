import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def squares_sum(w1, w2):
    return w1 ** 2 + w1 * w2 + w2 ** 2


w1 = np.arange(-2, 2, 0.1)
w2 = np.arange(-2, 2, 0.1)
w1, w2 = np.meshgrid(w1, w2)
L = squares_sum(w1, w2)

L = (L - L.min()) / (L.max() - L.min())  # normalization

colors = cm.Spectral(L)
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(w1, w2, L, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))

ax.set_xlabel('$w_0$')
ax.set_ylabel('$w_1$')
ax.set_zlabel('L')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()
