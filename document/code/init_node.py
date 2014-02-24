#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import os
import sys
import fitsio
import numpy as np
from transit import ldlc_simple

# Monkey patch the module search path to get the current version of george.
d = os.path.dirname
sys.path.insert(0, os.path.join(d(d(d(os.path.abspath(__file__))))))
import george

# Some constants.
tol, maxdepth = 0.1, 2  # Exposure time integration tolerance.

# Injected orbit parameters.
q1, q2, t0, tau, ror, b = 0.4, 0.3, 10., 0.5, 0.015, 0.5
period = 100.  # We're only going to have one transit so this just needs
               # to be a large number.


# Define the mean transit model.
def model(t, fstar, q1, q2, t0, tau, ror, b, texp=54.2 / 86400.):
    u1, u2 = 2*q1*q2, q1*(1-2*q2)
    lc = ldlc_simple(t, u1, u2, period, t0, tau, ror, b, texp, tol, maxdepth)
    return fstar * lc


# Load the data.
data = fitsio.read("kplr010593626-2011024051157_slc.fits")
t, f, fe, q = (data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"],
               data["SAP_QUALITY"])

# Mask missing data and normalize.
m = np.isfinite(f) * np.isfinite(t) * (q == 0)
t, f, fe, q = t[m], f[m], fe[m], q[m]
mu = np.median(f)
f, fe = f / mu, fe / mu
t -= np.min(t)

# Inject a transit.
p0 = np.array([-9.0, 0.2, -10.5, np.median(f), q1, q2, t0, tau, ror, b])
f *= model(t, 1, *(p0[4:]))


# Define the mean transit model.
def model(t, fstar, q1, q2, t0, tau, ror, b, texp=54.2 / 86400.):
    u1, u2 = 2*q1*q2, q1*(1-2*q2)
    lc = ldlc_simple(t, u1, u2, period, t0, tau, ror, b, texp, tol, maxdepth)
    return fstar * lc


# Define the probabilistic noise model.
def lnprior(lna, lns, lnd, fstar, q1, q2, t0, tau, ror, b):
    if not (0 < q1 < 1 and 0 < q2 < 1):
        return -np.inf
    if not 0 < ror < 1:
        return -np.inf
    if not 0 <= b <= 1.0:
        return -np.inf
    if not np.min(t) < t0 < np.max(t):
        return -np.inf
    return 0.0


def lnlike(lna, lns, lnd, fstar, q1, q2, t0, tau, ror, b):
    a, s, d2 = np.exp(lna), np.exp(lns), np.exp(2 * lnd)
    res = f - model(t, fstar, q1, q2, t0, tau, ror, b)
    gp = george.GaussianProcess([a, s], tol=1e-12, nleaf=40)
    gp.compute(t, fe + d2)
    ll = gp.lnlikelihood(res)

    # Check for failures.
    if not np.isfinite(ll):
        return -np.inf
    return ll


def lnprob(p):
    lp = lnprior(*p)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(*p)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
