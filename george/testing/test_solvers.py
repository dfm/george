# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_basic_solver"]

import numpy as np

from .. import kernels
from ..basic import BasicSolver
from ..hodlr import HODLRSolver


def _test_solver(Solver, N=100, seed=1234):
    # Set up the solver.
    kernel = 1e-4 * kernels.ExpSquaredKernel(1.0)
    solver = Solver(kernel)

    # Sample some data.
    np.random.seed(seed)
    x = np.random.randn(N, kernel.ndim)
    yerr = 1e-3 * np.ones(N)
    solver.compute(x, yerr)

    # Build the matrix.
    K = kernel.value(x)
    K[np.diag_indices_from(K)] += yerr ** 2

    # Check the determinant.
    sgn, lndet = np.linalg.slogdet(K)
    assert sgn == 1.0, "Invalid determinant"
    assert np.allclose(solver.log_determinant, lndet), "Incorrect determinant"

    # Check the inverse.
    assert np.allclose(solver.apply_inverse(K), np.eye(N)), "Incorrect inverse"


def test_basic_solver(**kwargs):
    _test_solver(BasicSolver, **kwargs)


def test_hodlr_solver(**kwargs):
    _test_solver(HODLRSolver, **kwargs)
