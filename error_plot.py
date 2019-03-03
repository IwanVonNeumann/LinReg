import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def squares_sum(w0, w1):
    return w0 ** 2 + w0 * w1 + w1 ** 2


w0, w1 = np.meshgrid(
    np.arange(-2, 2, 0.1),
    np.arange(-2, 2, 0.1)
)

J = squares_sum(w0, w1)
J = (J - J.min()) / (J.max() - J.min())  # normalization - to match colormap spectre

colors = cm.Spectral(J)
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(w0, w1, J, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0, 0, 0, 0))

ax.set_xlabel('$w_0$')
ax.set_ylabel('$w_1$')
ax.set_zlabel('J')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()
