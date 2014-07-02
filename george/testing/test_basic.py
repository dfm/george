#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["test_basic", "test_predict", "test_sample"]

import numpy as np

from .. import kernels
from ..george import HODLRGP
from ..basic import GaussianProcess


def test_basic(N=20, seed=123):
    kernel = 1e-6 * kernels.ExpKernel(1.0, 2)
    np.random.seed(seed)
    t = np.random.randn(N, kernel.ndim)
    y = np.sin(np.sum(t**2, axis=1))
    yerr = 1e-6 * np.ones_like(y)

    gp1 = GaussianProcess(kernel)
    gp1.compute(t, yerr)
    ll1 = gp1.lnlikelihood(y)

    gp2 = HODLRGP(kernel)
    gp2.compute(t, yerr)
    ll2 = gp2.lnlikelihood(y)

    assert np.allclose(ll1, ll2), (ll1, ll2)


def test_predict(N=20, N2=50, seed=123):
    kernel = 1e-6 * kernels.ExpKernel(1.0, 2)
    np.random.seed(seed)

    t = np.random.randn(N, kernel.ndim)
    t0 = np.random.randn(N2, kernel.ndim)

    y = np.sin(np.sum(t**2, axis=1))
    yerr = 1e-6 * np.ones_like(y)

    gp1 = GaussianProcess(kernel)
    gp1.compute(t, yerr)
    mu1, cov1 = gp1.predict(y, t0)

    gp2 = HODLRGP(kernel)
    gp2.compute(t, yerr)
    mu2, cov2 = gp2.predict(y, t0)

    assert np.allclose(mu1, mu2), "The predictive means don't match"
    assert np.allclose(cov1, cov2), \
        "The predictive covariances don't match"


def test_sample(N=20, N2=50, seed=123):
    kernel = 1e-6 * kernels.ExpKernel(1.0, 2)
    np.random.seed(seed)

    t = np.random.randn(N, kernel.ndim)
    t0 = np.random.randn(N2, kernel.ndim)

    y = np.sin(np.sum(t**2, axis=1))
    yerr = 1e-6 * np.ones_like(y)

    gp1 = GaussianProcess(kernel)
    gp1.compute(t, yerr)

    np.random.seed(seed)
    samps1 = gp1.sample_conditional(y, t0, 6)

    gp2 = HODLRGP(kernel)
    gp2.compute(t, yerr)

    np.random.seed(seed)
    samps2 = gp2.sample_conditional(y, t0, 6)

    assert np.allclose(samps1, samps2)
