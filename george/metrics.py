# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Metric", "Subspace"]

import fnmatch
import logging
import numpy as np
from scipy.linalg import cho_factor

from .compat import xrange


class Subspace(object):

    def __init__(self, ndim, axes=None):
        self.ndim = int(ndim)
        if axes is None:
            axes = np.arange(self.ndim)
        self.axes = np.atleast_1d(axes).astype(int)
        if np.any(self.axes >= self.ndim):
            raise ValueError("invalid axis for {0} dimensional metric"
                             .format(self.ndim))


class Metric(object):

    def __init__(self,
                 metric,
                 bounds=None,
                 ndim=None,
                 axes=None,
                 lower=True):
        if isinstance(metric, Metric):
            self.metric_type = metric.metric_type
            self.parameter_names = metric.parameter_names
            self.parameters = metric.parameters
            self.parameter_bounds = metric.parameter_bounds
            self.ndim = metric.ndim
            self.axes = metric.axes
            self.unfrozen = metric.unfrozen
            return

        if ndim is None:
            raise ValueError("missing required parameter 'ndim'")

        # Conform with the modeling protocol.
        self.parameter_names = []
        self.parameters = []
        self.parameter_bounds = None

        # Save the number of dimensions.
        subspace = Subspace(ndim, axes=axes)
        self.ndim = subspace.ndim
        self.axes = subspace.axes

        # See if the parameter is a scalar.
        try:
            metric = float(metric)

        except TypeError:
            metric = np.atleast_1d(metric)

            # If the metric is a vector, it is meant to be axis aligned.
            if len(metric.shape) == 1:
                self.metric_type = 1
                if len(metric) != len(self.axes):
                    raise ValueError("dimension mismatch")
                if np.any(metric <= 0.0):
                    raise ValueError("invalid (negative) metric")
                for i, v in enumerate(metric):
                    self.parameter_names.append("ln_M_{0}_{0}".format(i))
                    self.parameters.append(np.log(v))

                # Deal with bounded parameters.
                if bounds is not None:
                    if (len(bounds) != len(metric) or
                            not all(len(m) == 2 for m in bounds)):
                        raise ValueError("invalid bounds specification")
                    self.parameter_bounds = []
                    for i, v in enumerate(metric):
                        a, b = bounds[i]
                        self.parameter_bounds.append((
                            None if a is None else np.log(a),
                            None if b is None else np.log(b),
                        ))
                else:
                    self.parameter_bounds = [(None, None)
                                             for _ in xrange(len(metric))]

            elif len(metric.shape) == 2:
                self.metric_type = 2
                if metric.shape[0] != metric.shape[1]:
                    raise ValueError("metric must be square")
                if len(metric) != len(self.axes):
                    raise ValueError("dimension mismatch")

                # Compute the Cholesky factorization and log the diagonal.
                params = cho_factor(metric, lower=True)[0]
                i = np.diag_indices_from(params)
                params[i] = np.log(params[i])
                params = params[np.tril_indices_from(params)]

                # Save the parameter vector.
                k = 0
                for i in range(len(self.axes)):
                    self.parameter_names.append("ln_L_{0}_{0}".format(i))
                    self.parameters.append(params[k])
                    k += 1
                    for j in range(i+1, len(self.axes)):
                        self.parameter_names.append("L_{0}_{1}".format(i, j))
                        self.parameters.append(params[k])
                        k += 1

                # Bounded parameters?
                if bounds is not None:
                    logging.warn("bounded parameters are not implemented for "
                                 "the general metric")
                self.parameter_bounds = [(None, None)
                                         for _ in xrange(len(self.parameters))]

            else:
                raise ValueError("invalid metric dimensions")

        else:
            self.metric_type = 0
            self.parameter_names.append("ln_M_0_0")
            self.parameters.append(np.log(metric))

            # Deal with bounded parameters.
            if bounds is not None:
                if len(bounds) == 1:
                    bounds = bounds[0]
                if len(bounds) != 2:
                    raise ValueError("invalid bounds specification")
                a, b = bounds
                self.parameter_bounds = [(
                    None if a is None else np.log(a),
                    None if b is None else np.log(b),
                )]
            else:
                self.parameter_bounds = [(None, None)]

        self.parameters = np.array(self.parameters)
        self.unfrozen = np.ones_like(self.parameters, dtype=bool)

    def __len__(self):
        return np.sum(self.unfrozen)

    def get_parameter_names(self, full=False):
        if full:
            return self.parameter_names
        return [n for i, n in enumerate(self.parameter_names)
                if self.unfrozen[i]]

    def get_vector(self):
        return self.parameters[self.unfrozen]

    def check_vector(self, vector):
        n = 0
        for i, (a, b) in enumerate(self.parameter_bounds):
            if not self.unfrozen[i]:
                continue
            v = vector[n]
            if (a is not None and v < a) or (b is not None and b < v):
                return False
            n += 1
        return True

    def set_vector(self, vector):
        self.parameters[self.unfrozen] = vector

    def _match_parameter(self, k):
        names = self.parameter_names
        inds = [i for i, n in enumerate(names) if fnmatch.fnmatch(n, k)]
        if not len(inds):
            raise ValueError("unknown parameter '{0}'".format(k))
        return inds

    def freeze_parameter(self, parameter_name):
        self.unfrozen[self._match_parameter(parameter_name)] = False

    def thaw_parameter(self, parameter_name):
        self.unfrozen[self._match_parameter(parameter_name)] = True

    def get_parameter(self, parameter_name):
        p = self.parameters[self._match_parameter(parameter_name)]
        try:
            return float(p)
        except TypeError:
            return p

    def set_parameter(self, parameter_name, value):
        self.parameters[self._match_parameter(parameter_name)] = value

    def get_bounds(self):
        return [b for i, b in enumerate(self.parameter_bounds)
                if self.unfrozen[i]]

    def to_matrix(self):
        if self.metric_type == 0:
            return np.exp(self.parameters[0]) * np.eye(len(self.axes))
        elif self.metric_type == 1:
            return np.diag(np.exp(self.parameters))
        else:
            n = len(self.axes)
            L = np.zeros((n, n))
            L[np.tril_indices_from(L)] = self.parameters
            i = np.diag_indices_from(L)
            L[i] = np.exp(L[i])
            return np.dot(L, L.T)

    def __repr__(self):
        if self.metric_type == 0:
            params = ["{0}".format(float(np.exp(self.parameters)))]
        elif self.metric_type == 1:
            params = ["{0}".format(np.exp(self.parameters))]
        elif self.metric_type == 2:
            params = ["{0}".format(self.to_matrix().tolist())]
        params += ["ndim={0}".format(self.ndim), "axes={0}".format(self.axes)]
        params += ["bounds={0}".format([
            (None if a is None else np.exp(a),
             None if b is None else np.exp(b))
            for a, b in self.parameter_bounds
        ])]
        return "Metric({0})".format(", ".join(params))
