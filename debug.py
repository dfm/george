from __future__ import print_function

import numpy as np
from matplotlib import pyplot as pl

from george import GaussianProcess


np.random.seed(42)


def f(x):
    return x * np.sin(x)

X = 10 * np.random.rand(50)

# Observations
y = f(X).ravel()
yerr = np.ones_like(y)

x = np.atleast_2d(np.linspace(-5, 15, 1001)).T

gp = GaussianProcess([0.1, 1.0])
gp.optimize(X, y, yerr=yerr)

mu, var = gp.predict(x)

print(gp.evaluate())

std = np.sqrt(var.diagonal())

vals = np.random.multivariate_normal(mu, var, 100)

pl.plot(x, mu, "k")
pl.plot(x, vals.T, "k", alpha=0.1)

pl.plot(x, mu + std, ":r")
pl.plot(x, mu - std, ":r")

pl.errorbar(X, y, yerr=yerr, fmt=".r")
pl.savefig("debug.png")
