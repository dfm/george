#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import george
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl

from init_node import t, f, fe, model

p0, chain, lp, accept = pickle.load(open("results.pkl", "r"))

pl.plot(chain[:, :, 9].T)
pl.savefig("time.png")

assert 0


def prediction(times, lna, lns, lnd, fstar, q1, q2, t0, tau, ror, b, N=12):
    a, s = np.exp(lna), np.exp(lns)
    res = f - model(t, fstar, q1, q2, t0, tau, ror, b)
    gp = george.GaussianProcess([a, s], tol=1e-12, nleaf=40)
    gp.compute(t, fe)
    m2 = model(times, fstar, q1, q2, t0, tau, ror, b)
    samples = gp.sample_conditional(res, times, N=N)
    return samples + m2[None, :]


times = np.linspace(min(t), max(t), 500)
pl.plot(times, prediction(times, *p0).T, "k", alpha=0.5)
pl.savefig("initial.png")
