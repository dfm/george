# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HODLRSolver"]

import numpy as np

from .basic import BasicSolver
from ._hodlr import HODLRSolver as HODLRSolverInterface


class HODLRSolver(BasicSolver):

    def __init__(self, kernel, min_size=100, tol=0.1, seed=42):
        self.min_size = min_size
        self.tol = tol
        self.seed = seed
        super(HODLRSolver, self).__init__(kernel)

    def compute(self, x, yerr):
        self.solver = HODLRSolverInterface()
        self.solver.compute(self.kernel, x, yerr,
                            self.min_size, self.tol, self.seed)
        self._log_det = self.solver.log_determinant
        self.computed = self.solver.computed

    def apply_inverse(self, y, in_place=False):
        return self.solver.apply_inverse(y, in_place=in_place)

    def dot_solve(self, y):
        return self.solver.dot_solve(y)

    def apply_sqrt(self, r):
        raise NotImplementedError("apply_sqrt is not implemented for the "
                                  "HODLRSolver")

    def get_inverse(self):
        return self.solver.get_inverse()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_computed"] = False
        del state["solver"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
