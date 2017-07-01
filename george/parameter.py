# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ParameterVector"]

import numpy as np
from collections import OrderedDict


class Parameter(object):

    def __init__(self, name, value, log=True):
        self.log = log
        self.name = name
        self.frozen = False
        self.set_value(value, raw=True)

    def freeze(self):
        self.frozen = True

    def thaw(self):
        self.frozen = False

    def set_value(self, value, raw=False):
        if raw and self.log:
            self._value = np.log(value)
        else:
            self._value = value

    def get_value(self, raw=False):
        if raw and self.log:
            return np.exp(self._value)
        return self._value

    def __repr__(self):
        return "Parameter({0}, {1}, log={2})".format(
            self.name, self.get_value(raw=True), self.log
        )


class ParameterVector(object):

    parameters = OrderedDict()

    def freeze(self, key):
        self.parameters[key].freeze()

    def thaw(self, key):
        self.parameters[key].thaw()

    def __getattr__(self, attr):
        try:
            return self.parameters[attr]
        except KeyError:
            raise AttributeError("no attribute '{0}'".format(attr))

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        try:
            self.parameters[key].set_value(value, raw=True)
        except KeyError:
            if isinstance(value, Parameter):
                self.parameters[key] = value
            else:
                self.parameters[key] = Parameter(key, value)

    def get_parameter_vector(self, raw=False):
        return np.array([p.get_value(raw=raw)
                         for _, p in self.parameters.items()
                         if not p.frozen])

    def set_parameter_vector(self, vector, raw=False):
        i = 0
        for k, p in self.parameters.items():
            if p.frozen:
                continue
            p.set_value(vector[i], raw=raw)
            i += 1
        if i != len(vector):
            raise ValueError("dimension mismatch")
