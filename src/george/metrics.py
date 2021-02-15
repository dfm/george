# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Metric", "Subspace"]

import numpy as np
from scipy.linalg import cho_factor

from .modeling import Model


class Subspace(object):

    def __init__(self, ndim, axes=None):
        self.ndim = int(ndim)
        if axes is None:
            axes = np.arange(self.ndim)
        self.axes = np.atleast_1d(axes).astype(int)
        if np.any(self.axes >= self.ndim):
            raise ValueError("invalid axis for {0} dimensional metric"
                             .format(self.ndim))


class Metric(Model):

    def __init__(self,
                 metric,
                 bounds=None,
                 ndim=None,
                 axes=None,
                 lower=True):
        if isinstance(metric, Metric):
            self.metric_type = metric.metric_type
            self.parameter_names = metric.parameter_names
            self.unfrozen_mask = metric.unfrozen_mask
            self.set_parameter_vector(
                metric.get_parameter_vector(include_frozen=True),
                include_frozen=True)
            self.parameter_bounds = metric.parameter_bounds
            self.ndim = metric.ndim
            self.axes = metric.axes
            return

        if ndim is None:
            raise ValueError("missing required parameter 'ndim'")

        # Conform with the modeling protocol.
        parameter_names = []
        parameters = []

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
                    parameter_names.append("log_M_{0}_{0}".format(i))
                    parameters.append(np.log(v))
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
                    parameter_names.append("log_L_{0}_{0}".format(i))
                    parameters.append(params[k])
                    k += 1
                    for j in range(i+1, len(self.axes)):
                        parameter_names.append("L_{0}_{1}".format(i, j))
                        parameters.append(params[k])
                        k += 1
            else:
                raise ValueError("invalid metric dimensions")

        else:
            self.metric_type = 0
            parameter_names.append("log_M_0_0")
            parameters.append(np.log(metric))

        self.parameter_names = tuple(parameter_names)
        kwargs = dict(zip(parameter_names, parameters))
        if bounds is not None:
            kwargs["bounds"] = bounds
        super(Metric, self).__init__(**kwargs)

    def to_matrix(self):
        vector = self.get_parameter_vector(include_frozen=True)
        if self.metric_type == 0:
            return np.exp(vector) * np.eye(len(self.axes))
        elif self.metric_type == 1:
            return np.diag(np.exp(vector))
        else:
            n = len(self.axes)
            L = np.zeros((n, n))
            L[np.tril_indices_from(L)] = vector
            i = np.diag_indices_from(L)
            L[i] = np.exp(L[i])
            return np.dot(L, L.T)

    def __repr__(self):
        vector = self.get_parameter_vector(include_frozen=True)
        if self.metric_type == 0:
            params = ["{0}".format(float(np.exp(vector)))]
        elif self.metric_type == 1:
            params = ["{0}".format(repr(np.exp(vector)))]
        elif self.metric_type == 2:
            params = ["{0}".format(repr(self.to_matrix().tolist()))]
        params += ["ndim={0}".format(self.ndim),
                   "axes={0}".format(repr(self.axes))]
        params += ["bounds={0}".format([
            (None if a is None else np.exp(a),
             None if b is None else np.exp(b))
            for a, b in self.get_parameter_bounds(include_frozen=True)
        ])]
        return "Metric({0})".format(", ".join(params))
