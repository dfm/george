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

np.random.seed(12345)

kernel = ExpSquaredKernel([3, 0.5], ndim=2)
gp = george.HODLRGP(kernel, tol=1e-10)

x, y = np.linspace(-5, 5, 62), np.linspace(-5, 5, 60)
x, y = np.meshgrid(x, y, indexing="ij")
shape = x.shape
samples = np.vstack((x.flatten(), y.flatten())).T
gp.compute(samples, 1e-4*np.ones(len(samples)), sort=False)

print(len(samples))
i = george.utils.nd_sort_samples(samples)

img = gp.get_matrix(samples[i])
pl.imshow(img, cmap="gray", interpolation="nearest")
pl.gca().set_xticklabels([])
pl.gca().set_yticklabels([])
pl.colorbar()
pl.savefig("2d-cov.png")

pl.clf()
z = np.empty(len(samples))
z[i] = gp.sample(samples[i])
pl.pcolor(x, y, z.reshape(shape), cmap="gray")
pl.colorbar()
pl.savefig("2d.png")

import time

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z), sort=False)
print(time.time() - s)
s = time.time()
print(gp.lnlikelihood(z))
print(time.time() - s)

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z))
print(gp.lnlikelihood(z))
print(time.time() - s)

gp.kernel = ExpSquaredKernel([3.1, 0.6], ndim=2)

s = time.time()
gp.compute(samples, 1e-4*np.ones_like(z))
print(gp.lnlikelihood(z))
print(time.time() - s)

s = time.time()
gp.compute(samples[i], 1e-4*np.ones_like(z), sort=False)
print(gp.lnlikelihood(z[i]))
print(time.time() - s)
