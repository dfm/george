#!/usr/bin/env python
# -*- coding: utf-8 -*-

import george
import numpy as np
import matplotlib.pyplot as pl

np.random.seed(123)

N = 100
x = 10 * np.random.rand(N) - 5
yerr = 0.05 * np.ones(len(x))
y = np.exp(-0.5 * x ** 2) - 0.5 + yerr * np.random.randn(len(x))

gp = george.GaussianProcess([1.0, 1.0])
gp.compute(x, yerr)
print(gp.lnlikelihood(y))

ntest = 150
t = np.linspace(-6, 6, ntest)
mu, cov = gp.predict(y, t)

pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.plot(t, mu, "k")
std = np.sqrt(np.diag(cov))
pl.plot(t, mu + std, "--k")
pl.plot(t, mu - std, "--k")
pl.savefig("demo.png")
