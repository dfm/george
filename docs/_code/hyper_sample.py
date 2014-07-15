#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reproduction of Figure 5.6 from Rasmussen & Williams (2006).
http://www.gaussianprocess.org/gpml/

"""

from __future__ import division, print_function

import time
import emcee
import numpy as np
import cPickle as pickle
import statsmodels.api as sm
import matplotlib.pyplot as pl

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
    # Trivial improper prior: uniform in the log.
    lnprior = 0.0

    # Update the kernel and compute the lnlikelihood.
    kernel.pars = np.exp(p)
    try:
        return lnprior + gp.lnlikelihood(y)
    except np.linalg.LinAlgError:
        return -np.inf


# Set up the sampler and initialize the walkers.
nwalkers, ndim = 36, len(kernel)
p0 = [np.log(kernel.pars) + 1e-8 * np.random.randn(ndim)
      for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
print("Sampling...")
s = time.time()
sampler.run_mcmc(p0, 5)
print("Finished {0}".format(time.time() - s))

pickle.dump((sampler.chain, sampler.lnprobability, gp),
            open("hyper-results.pkl", "wb"), -1)

# # Plot the traces.
# samples = sampler.chain
# for i in range(samples.shape[2]):
#     pl.clf()
#     pl.plot(samples[:, :, i].T, color="k", alpha=0.5)
#     pl.savefig("time/{0:03d}.png".format(i))
