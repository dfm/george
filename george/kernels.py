#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Sum", "Product", "Kernel",
           "ConstantKernel", "DotProductKernel", "ExpKernel",
           "ExpSquaredKernel", "CosineKernel", "Matern32Kernel",
           "Matern52Kernel"]

import numpy as np


class Kernel(object):

    is_kernel = True
    kernel_type = -1

    def __init__(self, *pars):
        self.pars = np.array(pars)

    def __len__(self):
        return len(self.pars)

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(float(b)), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(float(b)), self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(Kernel):
    is_kernel = False
    operator_type = -1

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2


class Sum(_operator):
    is_kernel = False
    operator_type = 0


class Product(_operator):
    is_kernel = False
    operator_type = 1


class ConstantKernel(Kernel):
    kernel_type = 0

    def __init__(self, value):
        super(ConstantKernel, self).__init__(value)


class DotProductKernel(Kernel):
    kernel_type = 1

    def __init__(self):
        super(DotProductKernel, self).__init__()


class ExpKernel(Kernel):
    kernel_type = 2

    def __init__(self, scale):
        super(ExpKernel, self).__init__(scale)


class ExpSquaredKernel(Kernel):
    kernel_type = 3

    def __init__(self, scale):
        super(ExpSquaredKernel, self).__init__(scale)


class CosineKernel(Kernel):
    kernel_type = 4

    def __init__(self, period):
        super(CosineKernel, self).__init__(period)


class Matern32Kernel(Kernel):
    kernel_type = 5

    def __init__(self, scale):
        super(Matern32Kernel, self).__init__(scale)


class Matern52Kernel(Kernel):
    kernel_type = 6

    def __init__(self, scale):
        super(Matern52Kernel, self).__init__(scale)
