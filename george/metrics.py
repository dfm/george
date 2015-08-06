# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Metric", "Subspace"]

import numpy as np


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
        try:
            self.ndim = metric.ndim
            self.metric_type = metric.metric_type
            self.axes = metric.axes
            self.params = metric.params
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
                self.params = metric

            elif len(metric.shape) == 2:
                self.metric_type = 2
                if metric.shape[0] != metric.shape[1]:
                    raise ValueError("metric must be square")
                if len(metric) != len(self.axes):
                    raise ValueError("dimension mismatch")
                if lower:
                    self.params = metric[np.tril_indices_from(metric)]
                else:
                    self.params = metric[np.triu_indices_from(metric)]

            else:
                raise ValueError("invalid metric dimensions")

        else:
            self.params = np.atleast_1d(metric)
