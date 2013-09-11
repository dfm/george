#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["GaussianProcess"]

import numpy as np
from ._george import _george


class GaussianProcess(object):
    """
    This is the money object.

    :params pars:
        The hyperparameters of the covariance function. For now, this must be
        a list or vector of length 3: amplitude, standard deviation, and
        truncation radius.

    """

    kernels = {
        "truncated_squared_exp": 0,
    }

    def __init__(self, pars, kernel_type="truncated_squared_exp"):
        self._gp = _george(pars)

    def compute(self, x, yerr=None):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :params x: ``(nsamples, )`` or ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :params yerr: (optional) ``(nsamples, )``
            The uncertainties on the data points at coordinates ``x``.

        """
        if yerr is None:
            yerr = np.zeros(len(x))
        return self._gp.compute(x, yerr)

    def lnlikelihood(self, y):
        """
        Compute the log-likelihood of a set of observations under the Gaussian
        process model. You must call ``compute`` before this function.

        :param y: ``(nsamples, )``
            The observations at the coordinates provided in the ``compute``
            step.

        """
        return self._gp.lnlikelihood(y)

    def predict(self, y, t):
        """
        Compute the conditional predictive distribution of the model.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :returns mu: ``(ntest, )``
            The mean of the predictive distribution.

        :returns cov: ``(ntest, ntest)``
            The full covariance matrix of the predictive distribution.

        """
        return self._gp.predict(y, t)

    def sample_conditional(self, y, t, N=1):
        """
        Draw samples from the predictive conditional distribution.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param N: (optional)
            The number of samples to draw.

        :returns samples: ``(N, ntest)``
            A list of predictions at coordinates given by ``t``.

        """
        mu, cov = self.predict(y, t)
        return np.random.multivariate_normal(mu, cov, size=N)

    def sample_prior(self, t, N=1):
        """
        Draw samples from the prior distribution.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the model should be sampled.

        :param N: (optional)
            The number of samples to draw.

        :returns samples: ``(N, ntest)``
            A list of predictions at coordinates given by ``t``.

        """
        cov = self._gp.covariance(t)
        return np.random.multivariate_normal(np.zeros(len(t)), cov, size=N)
