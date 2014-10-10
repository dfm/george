# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "test_gradient", "test_prediction", "test_repeated_prediction_cache"
]

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


def _test_prediction(solver=BasicSolver):
    """Basic sanity checks for GP regression."""

    kernel = kernels.ExpSquaredKernel(1.0)
    gp = GP(kernel, solver=solver)

    x = np.array((-1, 0, 1))
    gp.compute(x)

    y = x/x.std()
    mu, cov = gp.predict(y, x)

    assert np.allclose(y, mu), \
        "GP must predict noise-free training data exactly ({0}).\n({1})" \
        .format(solver.__name__, y - mu)

    assert np.all(cov > -1e-15), \
        "Covariance matrix must be nonnegative ({0}).\n{1}" \
        .format(solver.__name__, cov)

    var = np.diag(cov)
    assert np.allclose(var, 0), \
        "Variance must vanish at noise-free training points ({0}).\n{1}" \
        .format(solver.__name__, var)

    t = np.array((-.5, .3, 1.2))
    var = np.diag(gp.predict(y, t)[1])
    assert np.all(var > 0), \
        "Variance must be positive away from training points ({0}).\n{1}" \
        .format(solver.__name__, var)


def test_prediction(**kwargs):
    _test_prediction(solver=BasicSolver, **kwargs)
    _test_prediction(solver=HODLRSolver, **kwargs)


def test_repeated_prediction_cache():
    kernel = kernels.ExpSquaredKernel(1.0)
    gp = GP(kernel)

    x = np.array((-1, 0, 1))
    gp.compute(x)

    t = np.array((-.5, .3, 1.2))

    y = x/x.std()
    mu0, mu1 = (gp.predict(y, t, mean_only=True) for _ in range(2))
    assert np.array_equal(mu0, mu1), \
        "Identical training data must give identical predictions " \
        "(problem with GP cache)."

    y2 = 2*y
    mu2 = gp.predict(y2, t, mean_only=True)
    assert not np.array_equal(mu0, mu2), \
        "Different training data must give different predictions " \
        "(problem with GP cache)."

    a0 = gp._alpha
    gp.kernel[0] += 0.1
    gp.recompute()
    gp._compute_alpha(y2)
    a1 = gp._alpha
    assert not np.allclose(a0, a1), \
        "Different kernel parameters must give different alphas " \
        "(problem with GP cache)."
