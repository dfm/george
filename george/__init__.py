#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "2.1.0-dev"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    # __all__ = ["GaussianProcess", "kernels"]
    # from .george import GaussianProcess
    from . import kernels
