# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
]

import pytest
import numpy as np

from george import kernels, GP
from george.solvers import hodlr

@pytest.mark.xfail
def test_custom_cholesky(seed=1234, ndim=5):
    np.random.seed(seed)

    # Build the matrix.
    L1 = np.random.randn(ndim, ndim)
    L1[np.diag_indices(ndim)] = np.exp(L1[np.diag_indices(ndim)])
    L1[np.triu_indices(ndim, 1)] = 0.0

    for L in (L1, np.eye(ndim)):
        A = np.dot(L, L.T)
        b = np.random.randn(ndim)

        Lvec = np.array(L)
        Lvec[np.diag_indices(ndim)] = 1. / Lvec[np.diag_indices(ndim)]

        Lb1 = np.linalg.solve(L, b)
        Lb2 = hodlr.custom_forward_sub(Lvec[np.tril_indices(ndim)],
                                       np.array(b))
        assert np.allclose(Lb1, Lb2)

        Ainvb1 = np.linalg.solve(A, b)
        Ainvb2 = hodlr.custom_backward_sub(Lvec[np.tril_indices(ndim)], Lb2)
        assert np.allclose(Ainvb1, Ainvb2)


def _general_metric(metric, N=100, ndim=3):
    kernel = 0.1 * kernels.ExpSquaredKernel(metric, ndim=ndim)

    x = np.random.rand(N, ndim)
    M0 = kernel.get_value(x)

    gp = GP(kernel)
    M1 = gp.get_matrix(x)
    assert np.allclose(M0, M1)

    # Compute the expected matrix.
    M2 = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            r = x[i] - x[j]
            r2 = np.dot(r, np.linalg.solve(metric, r))
            M2[i, j] = 0.1 * np.exp(-0.5*r2)

    if not np.allclose(M0, M2):
        print(M0)
        print()
        print(M2)
        print()
        print(M0 - M2)
        print()
        print(M0 / M2)

        L = np.linalg.cholesky(metric)
        i = N - 1
        j = N - 2
        r = x[j] - x[i]
        print(x[i], x[j])
        print("r = ", r)
        print("L.r = ", np.dot(L, r))
    assert np.allclose(M0, M2)


def test_general_metric(seed=1234, N=2, ndim=3):
    np.random.seed(seed)

    _general_metric(np.eye(ndim), N=N, ndim=ndim)

    L = np.random.randn(ndim, ndim)
    L[np.diag_indices(ndim)] = np.exp(L[np.diag_indices(ndim)])
    L[np.triu_indices(ndim, 1)] = 0.0
    metric = np.dot(L, L.T)
    _general_metric(metric, N=N, ndim=ndim)


def test_axis_algined_metric(seed=1234, N=100, ndim=3):
    np.random.seed(seed)

    kernel = 0.1 * kernels.ExpSquaredKernel(np.ones(ndim), ndim=ndim)

    x = np.random.rand(N, ndim)
    M0 = kernel.get_value(x)

    gp = GP(kernel)
    M1 = gp.get_matrix(x)
    assert np.allclose(M0, M1)

    # Compute the expected matrix.
    M2 = 0.1*np.exp(-0.5*np.sum((x[None, :, :] - x[:, None, :])**2, axis=-1))
    assert np.allclose(M0, M2)
