# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ConstantModel", "CallableModel"]

import numpy as np

from .modeling import ModelingMixin


class ConstantModel(ModelingMixin):

    def __init__(self, value):
        super(ConstantModel, self).__init__(value=float(value))

    def get_value(self, x):
        return self["value"] + np.zeros(len(x), dtype=np.float64)

    def get_gradient(self, x):
        return np.ones((1, len(x)), dtype=np.float64)


class CallableModel(ModelingMixin):

    def __init__(self, function, gradient=None):
        self.function = function
        self.gradient = gradient

    def __len__(self):
        return 0

    def get_parameter_names(self):
        return []

    def get_value(self, x):
        return self.function(x)

    def get_gradient(self, x):
        if self.gradient is not None:
            return self.gradient(x)
        return super(CallableModel, self).get_gradient(x)

    def get_vector(self):
        return np.empty(0)

    def set_vector(self, v):
        pass
