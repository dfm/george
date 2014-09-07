# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_gradient"]

import numpy as np

from .. import kernels, GP, BasicSolver, HODLRSolver


def _test_gradient(seed=123, N=100, ndim=3, eps=1.32e-4, solver=BasicSolver):
    np.random.seed(seed)

    # Set up the solver.
    kernel = (0.1 * kernels.ExpSquaredKernel(0.5, ndim)
              + kernels.WhiteKernel(1e-8, ndim))
    gp = GP(kernel, solver=solver)

    # Sample some data.
    x = np.random.rand(N, ndim)
    y = gp.sample(x)
    gp.compute(x)

    # Compute the initial gradient.
    grad0 = gp.grad_lnlikelihood(y)

    for i in range(len(kernel)):
        # Compute the centered finite difference approximation to the gradient.
        kernel[i] += eps
        lp = gp.lnlikelihood(y)
        kernel[i] -= 2*eps
        lm = gp.lnlikelihood(y)
        kernel[i] += eps
        grad = 0.5 * (lp - lm) / eps
        assert np.abs(grad - grad0[i]) < 5 * eps, \
            "Gradient computation failed in dimension {0} ({1})\n{2}" \
            .format(i, solver.__name__, np.abs(grad - grad0[i]))


def test_gradient(**kwargs):
    _test_gradient(solver=BasicSolver, **kwargs)
    _test_gradient(solver=HODLRSolver, **kwargs)
    assert 0
