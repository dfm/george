# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Metric", "Subspace"]

import copy
import numpy as np
from scipy.linalg import cho_factor

from .parameter import Parameter, ParameterVector


class Subspace(object):

    def __init__(self, ndim, axes=None):
        self.ndim = int(ndim)
        if axes is None:
            axes = np.arange(self.ndim)
        self.axes = np.atleast_1d(axes).astype(int)
        if np.any(self.axes >= self.ndim):
            raise ValueError("invalid axis for {0} dimensional metric"
                             .format(self.ndim))


class Metric(ParameterVector):

    def __init__(self, metric, ndim=None, axes=None, lower=True):
        try:
            self.ndim = metric.ndim
            self.metric_type = metric.metric_type
            self.axes = metric.axes
            self.params = metric.params
            self.parameter_names = metric.parameter_names
            return

        except AttributeError:
            if ndim is None:
                raise ValueError("missing required parameter 'ndim'")

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
                    self["m_{0}_{0}".format(i)] = v

            elif len(metric.shape) == 2:
                self.metric_type = 2
                if metric.shape[0] != metric.shape[1]:
                    raise ValueError("metric must be square")
                if len(metric) != len(self.axes):
                    raise ValueError("dimension mismatch")
                if lower:
                    params = metric[np.tril_indices_from(metric)]
                else:
                    params = metric[np.triu_indices_from(metric)]

                # Save the parameter vector.
                self.set_parameter_vector(params, raw=True)

            else:
                raise ValueError("invalid metric dimensions")

        else:
            self.metric_type = 0
            self["m_0_0"] = metric

    def set_parameter_vector(self, vector, raw=False):
        if self.metric_type == 2:
            if raw:
                k = 0
                for i in range(len(self.axes)):
                    for j in range(i, len(self.axes)):
                        n = "m_{0}_{1}".format(i, j)
                        self[n] = Parameter(n, vector[k], log=False)
                        k += 1

            else:
                y = self._to_cholesky(vector)
                m = np.dot(y, y.T)
                self.set_parameter_vector(m[np.tril_indices_from(m)],
                                          raw=True)

        else:
            super(Metric, self).set_parameter_vector(vector, raw=raw)

    def get_parameter_vector(self, raw=False):
        if raw or self.metric_type != 2:
            return super(Metric, self).get_parameter_vector(raw=raw)

        v = super(Metric, self).get_parameter_vector(raw=True)

        # Re-parameterize to Cholesky factorization.
        N = len(self.axes)
        m = np.empty((N, N))
        li = np.tril_indices_from(m)
        m[li] = v
        z = cho_factor(m, lower=True, overwrite_a=True)[0]
        di = np.diag_indices_from(z)
        z[di] = np.log(z[di])
        return z[li]

    def __repr__(self):
        if self.metric_type == 0:
            params = ["{0}".format(float(self.params))]
        elif self.metric_type == 1:
            params = ["{0}".format(self.params)]
        elif self.metric_type == 2:
            x = np.empty((len(self.axes), len(self.axes)))
            x[np.tril_indices_from(x)] = self.params
            x[np.triu_indices_from(x)] = self.params
            params = ["{0}".format(x)]
        params += ["ndim={0}".format(self.ndim), "axes={0}".format(self.axes)]
        return "Metric({0})".format(", ".join(params))

    def _to_cholesky(self, x):
        if self.metric_type != 2:
            raise RuntimeError("re-parameterization is only valid for general "
                               "metric")
        N = len(self.axes)
        y = np.zeros((N, N))
        y[np.tril_indices_from(y)] = x
        di = np.diag_indices_from(y)
        y[di] = np.exp(y[di])
        return y

    def log_jacobian(self, x=None):
        if x is None:
            x = self.get_parameter_vector()

        pass
