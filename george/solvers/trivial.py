# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["TrivialSolver"]

import numpy as np
from ..kernels import EmptyKernel


class TrivialSolver(object):

    def __init__(self, kernel=None):
        if (kernel is not None and
                kernel.kernel_type != EmptyKernel.kernel_type):
            raise ValueError("the trivial solver doesn't work with a kernel")
        self.computed = False
        self.log_determinant = None

    def compute(self, x, yerr):
        self._ivar = 1.0 / yerr ** 2
        self.log_determinant = 2 * np.sum(np.log(yerr))
        self.computed = True

    def apply_inverse(self, y, in_place=False):
        if not in_place:
            y = np.array(y)
        y[:] *= self._ivar
        return y

    def dot_solve(self, y):
        return np.sum(y**2 * self._ivar)

    def apply_sqrt(self, r):
        return r * np.sqrt(self._ivar)
