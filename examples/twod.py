from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time

import numpy as np
from matplotlib import pyplot as pl

from george import GaussianProcess


np.random.seed(42)


def f(x):
    return np.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2 / 0.1))


x = np.random.randn(200).reshape([100, 2])
y = f(x)
yerr = 0.01 * np.ones_like(y)

gp = GaussianProcess([0.1, 1.0, 1.0])
gp.optimize(x, y, yerr=yerr)

# Grid.
X, Y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
target = np.vstack([X.ravel(), Y.ravel()]).T

s = time.time()
mu, var = gp.predict(target)
print(time.time() - s)

s = time.time()
gp.predict(target, full_cov=False)[1]
print(time.time() - s)

ymin, ymax = y.min(), y.max()
colors = (y - ymin) / ymax

pl.figure(figsize=[4, 4])
samples = np.random.multivariate_normal(mu, var, 100)
for i in range(samples.shape[0]):
    pl.clf()
    pl.imshow((samples[i].reshape([40, 40]) - ymin) / ymax,
                                                    extent=[-3, 3, -3, 3])
    pl.scatter(x[:, 0], x[:, 1], c=colors)
    pl.xlim([-3, 3])
    pl.ylim([-3, 3])
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.savefig("2d/{0:04d}.png".format(i))
