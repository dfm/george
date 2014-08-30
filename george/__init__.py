# -*- coding: utf-8 -*-

__version__ = "0.2-dev"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["kernels", "GP", "BasicSolver"]

    from . import kernels
    from .gp import GP
    from .core import BasicSolver
