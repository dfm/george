# -*- coding: utf-8 -*-

__version__ = "0.3.1"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = [
        "kernels",
        "GP", "Metric",
        "TrivialSolver", "BasicSolver", "HODLRSolver",
    ]

    from . import kernels
    from .gp import GP
    from .metrics import Metric
    from .solvers import TrivialSolver, BasicSolver, HODLRSolver
