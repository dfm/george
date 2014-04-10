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
print(kernel.pars)
gp = george.GaussianProcess(kernel)

x, y = np.linspace(-5, 5, 42), np.linspace(-5, 5, 40)
x, y = np.meshgrid(x, y, indexing="ij")
shape = x.shape
samples = np.vstack((x.flatten(), y.flatten())).T
i = george.george.nd_sort_samples(samples)

# img = np.sum((samples[:, None, :] - samples[None, :, :]) ** 2, axis=2)
img = gp.get_matrix(samples[i])
pl.imshow(img, cmap="gray", interpolation="nearest")
pl.gca().set_xticklabels([])
pl.gca().set_yticklabels([])
pl.colorbar()
pl.savefig("2d-cov.png")

pl.clf()
z = np.empty(len(samples))
z[i] = gp.sample_prior(samples[i])
pl.pcolor(x, y, z.reshape(shape), cmap="gray")
pl.colorbar()
pl.savefig("2d.png")
