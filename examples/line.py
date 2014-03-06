#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as pl

d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
import george
from george.kernels import ExpSquaredKernel

np.random.seed(123)

true_kernel = 0.5 * ExpSquaredKernel(1.3)
true_gp = george.GaussianProcess(true_kernel)

# Generate some fake data.
true_m, true_b = 0.5, -0.25
N = 50
x = 10*np.sort(np.random.rand(N)) - 5
y = true_m * x + true_b
yerr = 0.1 + 0.5 * np.random.rand(N)

# Add some simulated noise into the data.
y += yerr * np.random.randn(N) + true_gp.sample_prior(x)

# Output the true values.
print("""True Values
===========
m = {0}
b = {1}
""".format(true_m, true_b))

# Do the least squares fit.
A = np.vander(x, 2)
Cinv = np.diag(1.0/yerr**2)
icov = np.dot(A.T, np.dot(Cinv, A))
ls_m, ls_b = np.linalg.solve(icov, np.dot(np.dot(Cinv, y), A))
ls_cov = np.linalg.inv(icov)
print("""Least Squares Result
====================
m = {0} ± {1}
b = {2} ± {3}
""".format(ls_m, np.sqrt(ls_cov[0, 0]), ls_b, np.sqrt(ls_cov[1, 1])))

# Compute the result with known covariance.
Cinv = np.linalg.inv(true_gp._gp.get_matrix(x) + np.diag(yerr**2))
icov = np.dot(A.T, np.dot(Cinv, A))
gp_m, gp_b = np.linalg.solve(icov, np.dot(np.dot(Cinv, y), A))
gp_cov = np.linalg.inv(icov)
print("""Known Covariance Result
=======================
m = {0} ± {1}
b = {2} ± {3}
""".format(gp_m, np.sqrt(gp_cov[0, 0]), gp_b, np.sqrt(gp_cov[1, 1])))

# Generate the area boundaries giving "posterior" constraints on the model.
x0 = np.vander(np.linspace(-5, 5, 100), 2)
samples = np.random.multivariate_normal([ls_m, ls_b], ls_cov, size=500)
values = np.dot(x0, samples.T)
ls_mu = np.mean(values, axis=1)
ls_std = np.std(values, axis=1)
samples = np.random.multivariate_normal([gp_m, gp_b], gp_cov, size=500)
values = np.dot(x0, samples.T)
gp_mu = np.mean(values, axis=1)
gp_std = np.std(values, axis=1)

# Plot the summary figure.
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, elinewidth=1, ms=6)

pl.fill_between(x0[:, 0], gp_mu+gp_std, gp_mu-gp_std, color="b", alpha=0.1)
pl.plot(x0[:, 0], gp_mu + gp_std, "-b", lw=1.5, alpha=0.5)
pl.plot(x0[:, 0], gp_mu - gp_std, "-b", lw=1.5, alpha=0.5)

pl.fill_between(x0[:, 0], ls_mu+ls_std, ls_mu-ls_std, color="r", alpha=0.1)
pl.plot(x0[:, 0], ls_mu + ls_std, "-r", lw=1.5, alpha=0.5)
pl.plot(x0[:, 0], ls_mu - ls_std, "-r", lw=1.5, alpha=0.5)

pl.plot(x0[:, 0], true_m * x0[:, 0] + true_b, "-k", lw=1.5)

pl.xlim(-5, 5)
pl.savefig("data.png")
