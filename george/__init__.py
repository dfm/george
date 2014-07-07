# -*- coding: utf-8 -*-

__version__ = "2.1.0-dev"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["kernels", "GP", "HODLRGP"]

    from . import kernels
    from .basic import GP
    from .hodlr import HODLRGP

    from testing import *
