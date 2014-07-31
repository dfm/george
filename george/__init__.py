# -*- coding: utf-8 -*-

__version__ = "0.1.1"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["kernels", "GP", "HODLRGP"]

    from . import kernels
    from .basic import GP
    from .hodlr import HODLRGP
