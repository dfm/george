#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys

d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as pl
from george import GaussianProcess
from george.kernels import ExpSquaredKernel, CosineKernel

np.random.seed(123)

kernel = ExpSquaredKernel(1.0, 2.3)
gp = GaussianProcess(kernel)

# Generate some fake data.
period = 0.956
x = 10 * np.sort(np.random.rand(75))
yerr = 0.1 + 0.1 * np.random.rand(len(x))
y = gp.sample_prior(x)
y += 0.8 * np.cos(2 * np.pi * x / period)
y += yerr * np.random.randn(len(yerr))

# Set up a periodic kernel.
pk = ExpSquaredKernel(np.sqrt(0.8), 1000.0) * CosineKernel(period)
kernel2 = kernel + pk
gp2 = GaussianProcess(kernel2)

# Condition on this data.
gp2.compute(x, yerr)

# Compute the log-likelihood.
print("Log likelihood = {0}".format(gp2.lnlikelihood(y)))

# Compute the conditional predictive distribution.
t = np.linspace(0, 10, 200)
f = gp2.sample_conditional(y, t, size=500)
mu = np.mean(f, axis=0)
std = np.std(f, axis=0)

pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.plot(t, mu, "k", lw=2, alpha=0.5)
pl.plot(t, mu+std, "k", alpha=0.5)
pl.plot(t, mu-std, "k", alpha=0.5)
pl.savefig("data.png")
