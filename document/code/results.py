#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import george
import triangle
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl

from init_node import t, f, fe, model

p0, chain, lp, accept = pickle.load(open("results.pkl", "r"))

flat = chain[:, 5000::13, :]
flat = flat.reshape((-1, flat.shape[-1]))
m = flat[:, 2] > -6.5
flat = flat[m]
labels = map(r"${0}$".format, [
    r"\ln \alpha", r"\ln l", r"\ln s", r"f_\star", r"q_1", r"q_2",
    r"t_0", r"\tau", r"r/R_\star", "b",
])
fig = triangle.corner(flat, bins=30, truths=p0, labels=labels,
                      quantiles=[0.16, 0.5, 0.84])
fig.savefig("corner.png")

assert 0


def prediction(times, lna, lns, lnd, fstar, q1, q2, t0, tau, ror, b):
    a, s = np.exp(lna), np.exp(lns)
    res = f - model(t, fstar, q1, q2, t0, tau, ror, b)
    gp = george.GaussianProcess([a, s], tol=1e-12, nleaf=40)
    gp.compute(t, fe)
    m2 = model(times, fstar, q1, q2, t0, tau, ror, b)
    mu, cov = gp.predict(res, times)
    return mu + m2


times = np.linspace(min(t), max(t), 750)
pl.figure()
pl.plot(t, f, ".k", alpha=0.3, ms=2)
for i in np.random.randint(len(flat), size=12):
    pl.plot(times, prediction(times, *flat[i]).T, "k", alpha=0.5)
pl.savefig("prediction.png")
