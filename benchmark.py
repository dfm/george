#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import george
import numpy as np
import matplotlib.pyplot as pl

np.random.seed(123)

N = 2000
x = 100 * np.sort(np.random.rand(N)) - 50
yerr = 0.05 * np.ones(len(x))
y = np.sin(0.5 * x) + yerr * np.random.randn(len(x))

gp = george.GaussianProcess([1.0, 2.0, 4.0])

strt = time.time()
gp.compute(x, yerr)
print(gp.lnlikelihood(y))
print(time.time() - strt)

ntest = 500
t = np.linspace(-60, 60, ntest)
mu, cov = gp.predict(y, t)

pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.plot(t, mu, "r")
std = np.sqrt(np.diag(cov))
pl.plot(t, mu + std, "-r")
pl.plot(t, mu - std, "-r")
pl.savefig("benchmark.png")
