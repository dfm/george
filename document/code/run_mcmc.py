#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import emcee
import numpy as np
import cPickle as pickle
from IPython.parallel import Client

from init_node import p0, lnprob

# Set up the interface to the ipcluster.
c = Client()
view = c[:]
view.push({"__file__": __file__}, block=True)
view.run("init_node.py", block=True)

# Set up the sampler.
ndim, nwalkers = len(p0), 32
pos = [p0 + 1e-10 * np.random.randn(ndim) for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=view)
for i, (p, lp, state) in enumerate(sampler.sample(pos, iterations=1000)):
    print(i)

# Save the results.
pickle.dump((p0, sampler.chain, sampler.lnprobability,
             sampler.acceptance_fraction),
            open("results.pkl", "wb"), -1)
