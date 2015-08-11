# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Metric", "Subspace"]

import numpy as np
from scipy.linalg import cho_factor


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

    def __init__(self, metric, ndim=None, axes=None, lower=True):
        if ndim is None:
            raise ValueError("missing required parameter 'ndim'")

        # Conform with the modeling protocol.
        self.parameter_names = []
        self.raw_parameters = []

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
                    self.parameter_names.append("ln(M_{0}_{0})".format(i))
                    self.raw_parameters.append(v)

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
                k = 0
                for i in range(len(self.axes)):
                    self.parameter_names.append("ln(L_{0}_{0})".format(i))
                    self.raw_parameters.append(params[k])
                    k += 1
                    for j in range(i+1, len(self.axes)):
                        self.parameter_names.append("L_{0}_{1}".format(i, j))
                        self.raw_parameters.append(params[k])
                        k += 1

            else:
                raise ValueError("invalid metric dimensions")

        else:
            self.metric_type = 0
            self.parameter_names.append("ln(M_0_0)")
            self.raw_parameters.append(metric)

        self.raw_parameters = np.array(self.raw_parameters)
        self.unfrozen = np.ones_like(self.raw_parameters, dtype=bool)

    def __len__(self):
        return np.sum(self.unfrozen)

    def get_parameter_names(self):
        return [n for i, n in enumerate(self.parameter_names)
                if self.unfrozen[i]]

    def get_vector(self):
        if self.metric_type == 2:
            pass

        else:
            return np.log(self.raw_parameters[self.unfrozen])

    def set_vector(self, vector):
        if self.metric_type == 2:
            pass

        else:
            self.raw_parameters[self.unfrozen] = np.exp(vector)

    def get_value(self, x1, x2):
        a, l = self.parameters
        r2 = (x1 - x2)**2
        return a * np.exp(-0.5 * r2 / l)

    def get_gradient(self, x1, x2):
        a, l = self.parameters
        value = self.get_value(x1, x2)
        grad = np.array((
            value,
            value * (0.5 * (x1 - x2)**2 / l)
        ))
        return grad[self.unfrozen]

    def freeze_parameter(self, parameter_name):
        names = self.get_parameter_names()
        self.unfrozen[names.index(parameter_name)] = False

    def thaw_parameter(self, parameter_name):
        names = self.get_parameter_names()
        self.unfrozen[names.index(parameter_name)] = True


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
        """
        For the general metric, we'll reparameterize the covariance matrix
        using a Cholesky decomposition.

        .. math::
            \Sigma = L \cdot L^\mathrm{T}

        and use the values of L (with logarithms of the diagonal).

        Refs: `farr/plotutils <https://github.com/farr/plotutils/blob/master/
        plotutils/parameterizations.py>`_ and `Stan <http://mc-stan.org/>`_.

        """
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
