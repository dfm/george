#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Sum", "Product", "Kernel",
           "ConstantKernel", "DotProductKernel", "ExpKernel",
           "ExpSquaredKernel", "RBFKernel", "CosineKernel", "ExpSine2Kernel",
           "Matern32Kernel", "Matern52Kernel"]

import numpy as np


class Kernel(object):

    is_kernel = True
    kernel_type = -1

    def __init__(self, *pars, **kwargs):
        self.ndim = kwargs.get("ndim", 1)
        self.pars = np.array(pars)

    def __len__(self):
        return len(self.pars)

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(np.sqrt(np.abs(float(b))),
                                      ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(np.sqrt(np.abs(float(b))),
                                          ndim=self.ndim), self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(Kernel):
    is_kernel = False
    operator_type = -1

    def __init__(self, k1, k2):
        if k1.ndim != k2.ndim:
            raise ValueError("Dimension mismatch")
        self.k1 = k1
        self.k2 = k2
        self.ndim = k1.ndim


class Sum(_operator):
    is_kernel = False
    operator_type = 0


class Product(_operator):
    is_kernel = False
    operator_type = 1


class ConstantKernel(Kernel):
    kernel_type = 0

    def __init__(self, value, ndim=1):
        super(ConstantKernel, self).__init__(value, ndim=ndim)


class DotProductKernel(Kernel):
    kernel_type = 1

    def __init__(self, ndim=1):
        super(DotProductKernel, self).__init__(ndim=ndim)


class _cov_kernel(Kernel):

    def __init__(self, cov, ndim=1):
        inds = np.tri(ndim, dtype=bool)
        try:
            l = len(cov)
        except TypeError:
            pars = np.diag(float(cov) * np.ones(ndim))[inds]
        else:
            if l == ndim:
                pars = np.diag(cov)[inds]
            else:
                pars = np.array(cov)
                if l != (ndim*ndim + ndim) / 2:
                    raise ValueError("Dimension mismatch")
        super(_cov_kernel, self).__init__(*pars, ndim=ndim)


class ExpKernel(_cov_kernel):
    kernel_type = 2


class ExpSquaredKernel(_cov_kernel):
    kernel_type = 3


class RBFKernel(_cov_kernel):
    kernel_type = 3


class CosineKernel(Kernel):
    kernel_type = 4

    def __init__(self, period, ndim=1):
        super(CosineKernel, self).__init__(period, ndim=ndim)


class ExpSine2Kernel(Kernel):
    kernel_type = 5

    def __init__(self, gamma, period, ndim=1):
        super(ExpSine2Kernel, self).__init__(gamma, period, ndim=ndim)


class Matern32Kernel(_cov_kernel):
    kernel_type = 6


class Matern52Kernel(_cov_kernel):
    kernel_type = 7
