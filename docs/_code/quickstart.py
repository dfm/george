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

# Draw 10k samples from the predictive conditional distribution.
t = np.linspace(0, 10, 500)
samples = gp.sample_conditional(y, t, size=10000)

# Compute the bounds of the prediction.
q = np.percentile(samples, [16.0, 84.0], axis=0)

pl.fill_between(t, q[0], q[1], color="k", alpha=0.2)
pl.errorbar(x, y, yerr=yerr, fmt="ok", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")
pl.savefig("../_static/quickstart/conditional.png")
