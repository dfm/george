#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = []

import numpy as np
import matplotlib.pyplot as pl

import transit

import george
from george.kernels import RBFKernel


def median_detrend(x, y, dt=1.):
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = np.abs(x-t) < 0.5 * dt
        r[i] = np.median(y[inds])
    return y / r


def model(p):
    q1, q2, per, t0, tau, ror, b = p
    u1, u2 = 2*q1*q2, q1*(1-2*q2)
    return transit.ldlc_simple(t, u1, u2, per, t0, tau, ror, b,
                               texp, 1e-1, 4)


texp = 1626.0 / 86400.0
true_params = [0.5, 0.5, 20, 5.0, 0.4, 0.015, 0.5]

gp = george.GaussianProcess(1e-8 * RBFKernel(0.9))

# Generate some fake data.
t = np.arange(0, 10.0, texp)
yerr = 2e-5 * np.ones_like(t)
print("sampling")
y = (gp.sample_prior(t) + 1) * model(true_params)
print("done")
y += yerr * np.random.randn(len(yerr))

pl.errorbar(t, y, yerr=yerr, fmt=".k")
pl.savefig("data.png")

print("computing")
gp.compute(t, yerr)
print("done")

print("lnlike")
print(gp.lnlikelihood(y - 1))
print(gp.lnlikelihood(y - model(true_params)))
print("done")

print("detrending")
y_detrend = median_detrend(t, y)
print("done")

pl.clf()
pl.errorbar(t, y_detrend, yerr=yerr, fmt=".k")
pl.savefig("detrend.png")
