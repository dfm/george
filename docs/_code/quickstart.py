#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

import george
from george.kernels import ExpSquaredKernel

np.random.seed(1234)

# Generate some fake noisy data.
x = 10 * np.sort(np.random.rand(10))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

# Set up the Gaussian process.
kernel = ExpSquaredKernel(1.0)
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

# Compute the log likelihood.
print(gp.lnlikelihood(y))

# Compute the predictive conditional distribution.
t = np.linspace(0, 10, 500)
mu, cov = gp.predict(y, t)
std = np.sqrt(np.diag(cov))

pl.fill_between(t, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(t, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu, color="k", alpha=1, lw=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")
pl.savefig("../_static/quickstart/conditional.png", dpi=150)
