from __future__ import print_function

import numpy as np
from matplotlib import pyplot as pl

from george import GaussianProcess


np.random.seed(1)


def f(x):
    return x * np.sin(x)

X = [1., 3., 5., 6., 7., 8.]

# Observations
y = f(X).ravel()
yerr = 0.1 * np.ones_like(y)

x = np.atleast_2d(np.linspace(-10, 20, 1001)).T

gp = GaussianProcess([0.1, 1.0])
mu, var, logprob = gp(X, y, x, yerr=yerr)

std = np.sqrt(var.diagonal())

vals = np.random.multivariate_normal(mu, var, 100)

pl.plot(x, mu, "k")
pl.plot(x, vals.T, "k", alpha=0.1)

pl.plot(x, mu + std, ":r")
pl.plot(x, mu - std, ":r")

pl.plot(X, y, ".r")
pl.savefig("debug.png")
