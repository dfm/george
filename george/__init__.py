# -*- coding: utf-8 -*-

__version__ = "0.2.1"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["kernels", "GP", "BasicSolver", "HODLRSolver"]

    from . import kernels
    from .gp import GP
    from .basic import BasicSolver
    from .hodlr import HODLRSolver
