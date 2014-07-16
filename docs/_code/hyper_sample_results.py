#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reproduction of Figure 5.6 from Rasmussen & Williams (2006).
http://www.gaussianprocess.org/gpml/

"""

from __future__ import division, print_function

import sys
import numpy as np
import cPickle as pickle
import statsmodels.api as sm
import matplotlib.pyplot as pl

# Load the dataset.
data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)

# Load the results.
chain, _, gp = pickle.load(open(sys.argv[1], "rb"))

# Set up the figure.
fig = pl.figure(figsize=(6, 3.5))
ax = fig.add_subplot(111)
ax.plot(t, y, ".k", ms=2)
ax.set_xlabel("year")
ax.set_ylabel("CO$_2$ in ppm")
fig.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.95)

# Plot the predictions.
x = np.linspace(max(t), 2025, 250)
for i in range(50):
    # Choose a random walker and step.
    w = np.random.randint(chain.shape[0])
    n = np.random.randint(2000, chain.shape[1])
    gp.kernel.pars = np.exp(chain[w, n])

    # Plot a single sample.
    ax.plot(x, gp.sample_conditional(y, x), "k", alpha=0.3)

ax.set_xlim(min(t), 2025.0)
ax.set_ylim(min(y), 420.0)
fig.savefig("../_static/hyper/mcmc.png", dpi=150)
