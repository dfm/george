#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = [
    "test_constant", "test_dot_prod",

    "test_exp", "test_exp_squared", "test_rbf", "test_matern32",
    "test_matern52",

    "test_cosine", "test_exp_sine2",

    "test_combine",
]

import numpy as np

from .. import kernels
from ..hodlr import HODLRGP


def do_kernel_t(kernel, N=20, seed=123):
    """
    Test that both the Python and C++ kernels return the same matrices.

    """
    np.random.seed(seed)
    t = np.random.randn(N, kernel.ndim)
    gp = HODLRGP(kernel)

    # Compute the two matrices.
    k1 = kernel(t[:, None], t[None, :])
    k2 = gp.get_matrix(t)

    # Build the matrix using brute force and check that.
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            v = kernel(a, b)
            assert np.allclose(k1[i, j], v), \
                "Python matrix fails (ndim = {2})\n{0} should be {1}" \
                .format(k1[i, j], v, kernel.ndim)
            assert np.allclose(k2[i, j], v), \
                "C++ matrix fails (ndim = {2})\n{0} should be {1}" \
                .format(k2[i, j], v, kernel.ndim)

    # Test that C++ and Python give the same matrix.
    assert np.allclose(k1, k2), (k1, k2)


#
# BASIC KERNELS
#

def test_constant():
    do_kernel_t(kernels.ConstantKernel(0.1))
    do_kernel_t(kernels.ConstantKernel(10.0, 2))
    do_kernel_t(kernels.ConstantKernel(5.0, 5))


def test_dot_prod():
    do_kernel_t(kernels.DotProductKernel())
    do_kernel_t(kernels.DotProductKernel(2))
    do_kernel_t(kernels.DotProductKernel(5))


#
# COVARIANCE KERNELS
#

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


def test_rbf():
    do_cov_t(kernels.RBFKernel)


def test_matern32():
    do_cov_t(kernels.Matern32Kernel)


def test_matern52():
    do_cov_t(kernels.Matern52Kernel)


#
# PERIODIC KERNELS
#

def test_cosine():
    do_kernel_t(kernels.CosineKernel(1.0))
    do_kernel_t(kernels.CosineKernel(0.5, 2))
    do_kernel_t(kernels.CosineKernel(0.75, 5))


def test_exp_sine2():
    do_kernel_t(kernels.ExpSine2Kernel(0.4, 1.0))
    do_kernel_t(kernels.ExpSine2Kernel(12.0, 0.5, 2))
    do_kernel_t(kernels.ExpSine2Kernel(13.7, 0.75, 5))


#
# COMBINING KERNELS
#

def test_combine():
    do_kernel_t(12 * kernels.ExpSine2Kernel(0.4, 1.0, ndim=5) + 0.1)
    do_kernel_t(12 * kernels.ExpSquaredKernel(0.4, ndim=3) + 0.1)
