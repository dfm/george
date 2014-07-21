#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["fit_gp"]

import os
import emcee
import numpy as np
import cPickle as pickle

from models import GPModel


def fit_gp(fn):
    # Load the data and set up the model.
    model = GPModel(fn)

    # Initialize the walkers.
    v = model.vector
    ndim, nwalkers = len(v), 32
    p0 = [v + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()

    print("Running production chain...")
    sampler.run_mcmc(p0, 4000)

    with open(os.path.splitext(fn)[0] + "-gp.pkl", "wb") as f:
        pickle.dump((model, sampler), f, -1)


if __name__ == "__main__":
    import sys
    fit_gp(sys.argv[1])
