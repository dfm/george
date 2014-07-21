#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_detrended"]

import os
import emcee
import numpy as np
import cPickle as pickle

from models import MedianModel


def fit_detrended(fn):
    # Load the data and set up the model.
    model = MedianModel(fn)

    # Initialize the walkers.
    v = model.vector
    ndim, nwalkers = len(v), 32
    p0 = [v + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production chain...")
    sampler.run_mcmc(p0, 2000)

    with open(os.path.splitext(fn)[0] + "-median.pkl", "wb") as f:
        pickle.dump((model, sampler), f, -1)


if __name__ == "__main__":
    import sys
    fit_detrended(sys.argv[1])
