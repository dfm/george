#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.0"

try:
    __GEORGE_SETUP__
except NameError:
    __GEORGE_SETUP__ = False

if not __GEORGE_SETUP__:
    __all__ = ["GaussianProcess"]
    from .george import GaussianProcess
    from . import kernels
