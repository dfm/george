#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from george import kernels

rad_temp = """    elif kernel_spec.kernel_type == {kernel_type}:
        if kernel_spec.dim >= 0:
            kernel = new {name}[OneDMetric](ndim,
                new OneDMetric(ndim, kernel_spec.dim))
        elif kernel_spec.isotropic:
            kernel = new {name}[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new {name}[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")
"""

for name, value in inspect.getmembers(kernels):
    if not hasattr(value, "__bases__"):
        continue
    if kernels.RadialKernel in value.__bases__:
        print(rad_temp.format(name=name, kernel_type=value.kernel_type))
