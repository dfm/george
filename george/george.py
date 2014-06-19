#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["GaussianProcess"]

import numpy as np
from ._george import _george

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def nd_sort_samples(samples):
    if cKDTree is None:
        raise ImportError("scipy.spatial.cKDTree is required for N-D sorting")

    # Check the shape of the sample list.
    assert len(samples.shape) == 2

    # Build a KD-tree on the samples.
    tree = cKDTree(samples)

    # Compute the distances.
    d, i = tree.query(samples[0], k=len(samples))
    return i


class GaussianProcess(object):
    """
    This is the money object.

    :params pars:
        The hyperparameters of the covariance function. For now, this must be
        a list or vector of length 2: amplitude and standard deviation.

    """

    def __init__(self, kernel, nleaf=100, tol=1e-12):
        self.nleaf = nleaf
        self.tol = tol
        self.kernel = kernel

    @property
    def computed(self):
        return self._gp.computed()

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        self._gp = _george(v, self.nleaf, self.tol)
        self._kernel = v

    def _parse_samples(self, t, sort):
        t = np.atleast_1d(t)
        if len(t.shape) == 1:
            # Deal with one-dimensional data.
            if sort:
                inds = np.argsort(t)
            else:
                inds = np.arange(len(t), dtype=int)
            t = np.atleast_2d(t).T
        elif sort:
            # Sort the data using a KD-tree.
            inds = nd_sort_samples(t)
        else:
            # Otherwise, assume that the samples are sorted.
            inds = np.arange(t.shape[0], dtype=int)

        # Double check the dimensions against the kernel.
        if len(t.shape) != 2 or t.shape[1] != self.kernel.ndim:
            raise ValueError("Dimension mismatch")

        return t[inds].T, inds

    def compute(self, x, yerr, sort=True):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :params x: ``(nsamples, )``
            The independent coordinates of the data points.

        :params yerr: ``(nsamples, )``
            The uncertainties on the data points at coordinates ``x``.

        """
        x0, self.inds = self._parse_samples(np.array(x), sort)
        return self._gp.compute(x0, yerr[self.inds])

    def lnlikelihood(self, y):
        """
        Compute the log-likelihood of a set of observations under the Gaussian
        process model. You must call ``compute`` before this function.

        :param y: ``(nsamples, )``
            The observations at the coordinates provided in the ``compute``
            step.

        """
        if not self.computed:
            raise RuntimeError("You need to compute the model first")
        ll = self._gp.lnlikelihood(y[self.inds])
        if np.isfinite(ll):
            return ll
        return -np.inf

    def predict(self, y, t):
        """
        Compute the conditional predictive distribution of the model.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )``
            The coordinates where the predictive distribution should be
            computed.

        :returns mu: ``(ntest, )``
            The mean of the predictive distribution.

        :returns cov: ``(ntest, ntest)``
            The predictive covariance.

        """
        t, i = self._parse_samples(t, False)
        return self._gp.predict(y, t)

    def sample_conditional(self, y, t, N=1, size=None, sort=True):
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
        if size is not None:
            N = size
        mu, cov = self.predict(y, t)
        samples = np.random.multivariate_normal(mu, cov, size=N)
        if N == 1:
            return samples[0]
        return samples

    def sample_prior(self, t, N=1, size=None):
        """
        Draw samples from the prior distribution.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the model should be sampled.

        :param N: (optional)
            The number of samples to draw.

        :returns samples: ``(N, ntest)``
            A list of predictions at coordinates given by ``t``.

        """
        if size is not None:
            N = size
        cov = self.get_matrix(t)
        samples = np.random.multivariate_normal(np.zeros(len(t)), cov, size=N)
        if N == 1:
            return samples[0]
        return samples

    def get_matrix(self, t):
        t, i = self._parse_samples(t, False)
        return self._gp.get_matrix(t)
