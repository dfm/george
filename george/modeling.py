# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ModelingMixin", "supports_modeling_protocol",
           "check_gradient"]

import numpy as np
from collections import OrderedDict

from .compat import iteritems, izip

_EPS = 1.254e-5


class ModelingMixin(object):

    _parameters = OrderedDict()
    _frozen = dict()

    def __init__(self, **kwargs):
        self._parameters = OrderedDict(sorted(iteritems(kwargs)))
        self._frozen = dict((k, False) for k in self._parameters)

    def __getitem__(self, k):
        try:
            i = int(k)
        except ValueError:
            return self._parameters[k]
        return self._parameters.values()[i]

    def __setitem__(self, k, v):
        try:
            i = int(k)
        except ValueError:
            pass
        else:
            k = self._parameters.keys()[i]

        if k in self._parameters:
            self._parameters[k] = v
        else:
            self._parameters[k] = v
            self._frozen[k] = False

    def __len__(self):
        return len(self._frozen) - sum(self._frozen.values())

    def get_parameter_names(self):
        return [k for k in self._parameters if not self._frozen[k]]

    def get_vector(self):
        return np.array([v for k, v in iteritems(self._parameters)
                         if not self._frozen[k]], dtype=np.float64)

    def set_vector(self, vector):
        for k, v in izip(self.get_parameter_names(), vector):
            self[k] = v

    def get_value(self, *args, **kwargs):
        raise NotImplementedError("'get_value' must be implemented by "
                                  "subclasses")

    def get_gradient(self, *args, **kwargs):
        vector = self.get_vector()
        value0 = self.get_value(*args, **kwargs)
        grad = np.empty([len(vector)] + list(value0.shape), dtype=np.float64)
        for i, v in enumerate(vector):
            vector[i] = v + _EPS
            self.set_vector(vector)
            value = self.get_value(*args, **kwargs)
            vector[i] = v
            self.set_vector(vector)
            grad[i] = (value - value0) / _EPS
        return grad

    def freeze_parameter(self, parameter_name):
        self._frozen[parameter_name] = True

    def thaw_parameter(self, parameter_name):
        self._frozen[parameter_name] = False


def supports_modeling_protocol(obj):
    # The modeling protocol requires the object to have a length.
    try:
        len(obj)
    except TypeError:
        return False

    # Check that all of the methods are implemented.
    methods = [
        "get_value",
        "get_gradient",
        "get_parameter_names",
        "get_vector",
        "set_vector",
        "freeze_parameter",
        "thaw_parameter",
    ]
    for method in methods:
        if not callable(getattr(obj, method, None)):
            return False
    return True


def check_gradient(obj, *args, **kwargs):
    eps = kwargs.pop("eps", 1.23e-5)

    grad0 = obj.get_gradient(*args, **kwargs)
    vector = obj.get_vector()
    for i, v in enumerate(vector):
        # Compute the centered finite difference approximation to the gradient.
        vector[i] = v + eps
        obj.set_vector(vector)
        p = obj.get_value(*args, **kwargs)

        vector[i] = v - eps
        obj.set_vector(vector)
        m = obj.get_value(*args, **kwargs)

        vector[i] = v
        obj.set_vector(vector)

        grad = 0.5 * (p - m) / eps
        assert np.allclose(grad0[i], grad), \
            "grad computation failed for '{0}' ({1})" \
            .format(obj.get_parameter_names()[i], i)
