#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reproduction of Figure 5.6 from Rasmussen & Williams (2006).
http://www.gaussianprocess.org/gpml/

"""

from __future__ import division, print_function

import sys
import time
import emcee
import numpy as np
import cPickle as pickle
import statsmodels.api as sm
import matplotlib.pyplot as pl
from multiprocessing import Pool

import george
from george import kernels

# Load the dataset.
data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)

# Initialize the kernel.
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) \
    * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
kernel = k1 + k2 + k3 + k4

# Set up the Gaussian process.
gp = george.GP(kernel, mean=np.mean(y))

s = time.time()
gp.compute(t)
gp.lnlikelihood(y)
print(time.time() - s)


# Define the probabilistic model.
def lnprob(p):
    # Trivial prior: uniform in the log.
    if np.any((-10 > p) + (p > 10)):
        return -np.inf
    lnprior = 0.0

    # Update the kernel and compute the lnlikelihood.
    kernel.pars = np.exp(p)
    return lnprior + gp.lnlikelihood(y, quiet=True)


# Set up the sampler and initialize the walkers.
nwalkers, ndim = 36, len(kernel)

if len(sys.argv) > 1:
    chain, _, _ = pickle.load(open(sys.argv[1], "rb"))
    p0 = chain[:, -1, :]
else:
    p0 = [np.log(kernel.pars) + 1e-4 * np.random.randn(ndim)
          for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=Pool())
print("Sampling...")
s = time.time()
for i in range(20):
    print(i+1)
    p0, _, _ = sampler.run_mcmc(p0, 200)
    pickle.dump((sampler.chain, sampler.lnprobability, gp),
                open("hyper-results-{0}.pkl".format(i), "wb"), -1)

print("Finished {0}".format(time.time() - s))
