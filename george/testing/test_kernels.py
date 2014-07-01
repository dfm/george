#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np

from .. import kernels
from .. import GaussianProcess


def do_kernel_t(kernel, N=50, seed=123):
    """
    Test that both the Python and C++ kernels return the same matrices.

    """
    np.random.seed(seed)
    t = np.random.randn(N, kernel.ndim)
    gp = GaussianProcess(kernel)

    k1 = kernel(t[:, None], t[None, :])
    k2 = gp.get_matrix(t)

    assert np.allclose(k1, k2), (k1, k2)


def do_cov_t(kernel_type):
    kernel = kernel_type(0.1)
    do_kernel_t(kernel)

    kernel = kernel_type(1.0)
    do_kernel_t(kernel)

    kernel = kernel_type(10.0)
    do_kernel_t(kernel)

    m = [1.0,
         0.5, 2.0,
         0.1, 0.3, 0.7]
    kernel = kernel_type(m, ndim=3)
    assert np.allclose(kernel.matrix, [[1.0, 0.5, 0.1],
                                       [0.5, 2.0, 0.3],
                                       [0.1, 0.3, 0.7]])
    do_kernel_t(kernel)

    kernel = kernel_type([1.0, 0.1, 10.0], ndim=3)
    do_kernel_t(kernel)

    kernel = kernel_type(1.0, ndim=3)
    do_kernel_t(kernel)

    try:
        kernel = kernel_type([1.0, 0.1, 10.0, 500], ndim=3)
    except ValueError:
        pass
    else:
        assert False, "This test should fail"


def test_exp():
    do_cov_t(kernels.ExpKernel)


def test_exp_squared():
    do_cov_t(kernels.ExpSquaredKernel)


def test_matern32():
    do_cov_t(kernels.Matern32Kernel)


def test_matern52():
    do_cov_t(kernels.Matern52Kernel)
