# -*- coding: utf-8 -*-

__version__ = "0.1.0"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["kernels", "GP", "HODLRGP"]

    from . import kernels
    from .basic import GP
    from .hodlr import HODLRGP

    # Tests.
    from .testing.test_basic import *  # NOQA
    from .testing.test_kernels import *  # NOQA
