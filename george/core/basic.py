# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BasicSolver"]

import numpy as np
from scipy.linalg import cholesky, cho_solve


class BasicSolver(object):

    def __init__(self, kernel):
        self.kernel = kernel

    def compute(self, x, yerr):
        # Compute the kernel matrix.
        K = self.kernel.value(x)
        K[np.diag_indices_from(K)] += yerr ** 2

        # Factor the matrix and compute the log-determinant.
        self._factor = (cholesky(K, overwrite_a=True, lower=False), False)
        self._lndet = None

    @property
    def log_determinant(self):
        if self._lndet is None:
            self._lndet = 2 * np.sum(np.log(np.diag(self._factor[0])))
        return self._lndet

    def apply_inverse(self, y, in_place=False):
        return cho_solve(self._factor, y, overwrite_b=in_place)
