# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "test_gradient", "test_prediction", "test_repeated_prediction_cache",
    "test_apply_inverse",
]

import pytest
import numpy as np
from itertools import product

from george import kernels, GP, BasicSolver, HODLRSolver

@pytest.mark.parametrize("solver,white_noise",
                         product([BasicSolver, HODLRSolver], [None, 0.1]))
def test_gradient(solver, white_noise, seed=123, N=305, ndim=3, eps=1.32e-3):
    np.random.seed(seed)

    # Set up the solver.
    kernel = 1.0 * kernels.ExpSquaredKernel(0.5, ndim=ndim)
    kwargs = dict()
    if white_noise is not None:
        kwargs = dict(white_noise=white_noise, fit_white_noise=True)
    if solver == HODLRSolver:
        kwargs["tol"] = 1e-8
    gp = GP(kernel, solver=solver, **kwargs)

    # Sample some data.
    x = np.random.rand(N, ndim)
    x = x[np.argsort(x[:, 0])]
    y = gp.sample(x)
    gp.compute(x, yerr=0.1)

    # Compute the initial gradient.
    grad0 = gp.grad_log_likelihood(y)
    vector = gp.get_parameter_vector()

    for i, v in enumerate(vector):
        # Compute the centered finite difference approximation to the gradient.
        vector[i] = v + eps
        gp.set_parameter_vector(vector)
        lp = gp.lnlikelihood(y)

        vector[i] = v - eps
        gp.set_parameter_vector(vector)
        lm = gp.lnlikelihood(y)

        vector[i] = v
        gp.set_parameter_vector(vector)

        grad = 0.5 * (lp - lm) / eps
        assert np.abs(grad - grad0[i]) < 5 * eps, \
            "Gradient computation failed in dimension {0} ({1})\n{2}" \
            .format(i, solver.__name__, np.abs(grad - grad0[i]))


@pytest.mark.parametrize("solver", [BasicSolver, HODLRSolver])
def test_prediction(solver, seed=42):
    """Basic sanity checks for GP regression."""

    np.random.seed(seed)

    kernel = kernels.ExpSquaredKernel(1.0)
    kwargs = dict()
    if solver == HODLRSolver:
        kwargs["tol"] = 1e-8
    gp = GP(kernel, solver=solver, white_noise=0.0, **kwargs)

    x0 = np.linspace(-10, 10, 500)
    x = np.sort(np.random.uniform(-10, 10, 300))
    gp.compute(x)

    y = np.sin(x)
    mu, cov = gp.predict(y, x0)

    Kstar = gp.get_matrix(x0, x)
    K = gp.get_matrix(x)
    K[np.diag_indices_from(K)] += 1.0
    mu0 = np.dot(Kstar, np.linalg.solve(K, y))
    print(np.abs(mu - mu0).max())
    assert np.allclose(mu, mu0)


def test_repeated_prediction_cache():
    kernel = kernels.ExpSquaredKernel(1.0)
    gp = GP(kernel)

    x = np.array((-1, 0, 1))
    gp.compute(x)

    t = np.array((-.5, .3, 1.2))

    y = x/x.std()
    mu0, mu1 = (gp.predict(y, t, return_cov=False) for _ in range(2))
    assert np.array_equal(mu0, mu1), \
        "Identical training data must give identical predictions " \
        "(problem with GP cache)."

    y2 = 2*y
    mu2 = gp.predict(y2, t, return_cov=False)
    assert not np.array_equal(mu0, mu2), \
        "Different training data must give different predictions " \
        "(problem with GP cache)."

    a0 = gp._alpha
    gp.kernel[0] += 0.1
    gp.recompute()
    gp._compute_alpha(y2, True)
    a1 = gp._alpha
    assert not np.allclose(a0, a1), \
        "Different kernel parameters must give different alphas " \
        "(problem with GP cache)."

    mu, cov = gp.predict(y2, t)
    _, var = gp.predict(y2, t, return_var=True)
    assert np.allclose(np.diag(cov), var), \
        "The predictive variance must be equal to the diagonal of the " \
        "predictive covariance."


@pytest.mark.parametrize("solver", [BasicSolver, HODLRSolver])
def test_apply_inverse(solver, seed=1234, N=201, yerr=0.1):
    np.random.seed(seed)

    # Set up the solver.
    kernel = 1.0 * kernels.ExpSquaredKernel(0.5)
    kwargs = dict()
    if solver == HODLRSolver:
        kwargs["tol"] = 1e-10
    gp = GP(kernel, solver=solver, **kwargs)

    # Sample some data.
    x = np.sort(np.random.rand(N))
    y = gp.sample(x)
    gp.compute(x, yerr=yerr)

    K = gp.get_matrix(x)
    K[np.diag_indices_from(K)] += yerr**2

    b1 = np.linalg.solve(K, y)
    b2 = gp.apply_inverse(y)
    assert np.allclose(b1, b2)

    y = gp.sample(x, size=5).T
    b1 = np.linalg.solve(K, y)
    b2 = gp.apply_inverse(y)
    assert np.allclose(b1, b2)


@pytest.mark.parametrize("solver", [BasicSolver, HODLRSolver])
def test_predict_single(solver, seed=1234, N=201, yerr=0.1):
    np.random.seed(seed)

    # Set up the solver.
    kernel = 1.0 * kernels.ExpSquaredKernel(0.5)
    kwargs = dict()
    if solver == HODLRSolver:
        kwargs["tol"] = 1e-8
    gp = GP(kernel, solver=solver, **kwargs)

    x = np.sort(np.random.rand(N))
    y = gp.sample(x)
    gp.compute(x, yerr=yerr)
    mu0, var0 = gp.predict(y, [0.0], return_var=True)
    mu, var = gp.predict(y, [0.0, 1.0], return_var=True)
    _, cov = gp.predict(y, [0.0, 1.0])
    assert np.allclose(mu0, mu[0])
    assert np.allclose(var0, var[0])
    assert np.allclose(var0, cov[0, 0])
