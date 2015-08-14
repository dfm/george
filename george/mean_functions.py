# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ConstantMean", "CallableMean"]

import numpy as np

from .modeling import ModelingMixin


class ConstantMean(ModelingMixin):

    def __init__(self, value):
        super(ConstantMean, self).__init__(value=float(value))

    def get_value(self, x):
        return self["value"] + np.zeros(len(x), dtype=np.float64)

    def get_gradient(self, x):
        return np.ones((1, len(x)), dtype=np.float64)


class CallableMean(ModelingMixin):

    def __init__(self, function):
        self.function = function

    def __len__(self):
        return 0

    def get_parameter_names(self):
        return []

    def get_value(self, x):
        return self.function(x)

    def get_vector(self):
        return np.empty(0)

    def set_vector(self, v):
        pass
