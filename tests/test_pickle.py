# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = [
    "test_pickle", "test_pickle",
]

import sys
import pytest
import pickle
import numpy as np

from george import GP, kernels, BasicSolver, HODLRSolver


def _fake_compute(arg, *args, **kwargs):
    assert 0, "Unpickled GP shouldn't need to be computed"


@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3")
@pytest.mark.parametrize("solver,success", [(BasicSolver, True),
                                            (HODLRSolver, False)])
def test_pickle(solver, success, N=50, seed=123):
    np.random.seed(seed)
    kernel = 0.1 * kernels.ExpSquaredKernel(1.5)
    kernel.pars = [1, 2]
    gp = GP(kernel, solver=solver)
    x = np.random.rand(100)
    gp.compute(x, 1e-2)

    s = pickle.dumps(gp, -1)
    gp = pickle.loads(s)
    if success:
        gp.compute = _fake_compute
    gp.lnlikelihood(np.sin(x))
