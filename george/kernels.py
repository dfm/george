#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Sum", "Product",
           "ExpSquaredKernel", "ExpKernel", "CosineKernel", "SparseKernel"]

import numpy as np


class _kernel(object):

    is_kernel = True
    kernel_type = -1

    def __init__(self, *pars):
        self.pars = np.array(pars)

    def __len__(self):
        return len(self.pars)


class _operator(object):

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


class ExpSquaredKernel(_kernel):

    kernel_type = 0

    def __init__(self, amplitude, scale):
        super(ExpSquaredKernel, self).__init__(amplitude, scale)


class ExpKernel(_kernel):

    kernel_type = 1

    def __init__(self, amplitude, scale):
        super(ExpKernel, self).__init__(amplitude, scale)


class CosineKernel(_kernel):

    kernel_type = 2

    def __init__(self, period):
        super(CosineKernel, self).__init__(period)


class SparseKernel(_kernel):

    kernel_type = 3

    def __init__(self, full_width):
        super(SparseKernel, self).__init__(full_width)
