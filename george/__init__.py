# -*- coding: utf-8 -*-

__version__ = "1.0.0.dev0"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = [
        "kernels",
        "GP", "Metric",
        "TrivialSolver", "BasicSolver", "HODLRSolver",
        "ModelingMixin",
    ]

    from . import kernels
    from .gp import GP
    from .metrics import Metric

    from .solvers import TrivialSolver, BasicSolver, HODLRSolver

    from .modeling import ModelingMixin
