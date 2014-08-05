#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np
import matplotlib.pyplot as pl

import george
from george import kernels

N = 2000
dt = 0.5 / 24.
x = np.linspace(0, N*dt, N)
y = np.sin(x)

kernel = 1.0 * kernels.Matern32Kernel(2.0)
gp = george.GP(kernel)

strt = time.time()
gp.compute(x, 0.1)
ll = gp.lnlikelihood(y)
print(time.time() - strt)
print(ll)

gp = george.HODLRGP(kernel)
strt = time.time()
gp.compute(x, 0.1)
ll = gp.lnlikelihood(y)
print(time.time() - strt)
print(ll)
