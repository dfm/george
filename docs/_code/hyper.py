#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as pl

import george
from george import kernels

# Load the dataset.
data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)
base = np.mean(y)

# Set up the Gaussian process.
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) \
    * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
kernel = k1 + k2 + k3 + k4

gp = george.GP(kernel, mean=base)
# gp.compute(t)
p, code = gp.optimize(t, y)
print(p)
print(gp.lnlikelihood(y))

x = np.linspace(max(t), 2025, 2000)
mu, cov = gp.predict(y, x)
std = np.sqrt(np.diag(cov))

fig = pl.figure(figsize=(6, 3.5))
ax = fig.add_subplot(111)
ax.fill_between(x, mu+std, mu-std, color="k", alpha=0.4)
ax.plot(t, y, ".k", ms=2)
ax.set_xlim(min(t), 2025.0)
ax.set_ylim(min(y), 420.0)
ax.set_xlabel("year")
ax.set_ylabel("CO$_2$ in ppm")
fig.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.95)

pl.savefig("data.png", dpi=150)
