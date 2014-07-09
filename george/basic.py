# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .utils import multivariate_gaussian_samples, nd_sort_samples


class GP(object):

    def __init__(self, kernel):
        self.kernel = kernel
        self._computed = False

    @property
    def computed(self):
        return self._computed and not self.kernel.dirty

    @computed.setter
    def computed(self, v):
        self._computed = v
        if v:
            self.kernel.dirty = False

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

        return t[inds], inds

    def _check_dimensions(self, y):
        n, ndim = self._x.shape
        y = np.atleast_1d(y)
        if len(y.shape) > 1:
            raise ValueError("The predicted dimension must be 1-D")
        if len(y) != n:
            raise ValueError("Dimension mismatch")
        return y

    def compute(self, x, yerr, sort=True, _scale=0.5*np.log(2*np.pi)):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :params x: ``(nsamples, )``
            The independent coordinates of the data points.

        :params yerr: ``(nsamples, )``
            The uncertainties on the data points at coordinates ``x``.

        """
        # Parse the input coordinates.
        self._x, self.inds = self._parse_samples(np.array(x), sort)
        self._yerr = self._check_dimensions(yerr)[self.inds]

        # Compute the kernel matrix.
        K = self.kernel(self._x[:, None], self._x[None, :])
        K[np.diag_indices_from(K)] += self._yerr ** 2

        # Factor the matrix and compute the log-determinant.
        factor, _ = self._factor = cho_factor(K, overwrite_a=True)
        self._const = -(np.sum(np.log(np.diag(factor))) + _scale*len(self._x))

        # Save the computed state.
        self.computed = True

    def _compute_lnlike(self, r):
        return self._const - 0.5*np.dot(r.T, cho_solve(self._factor, r))

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
        ll = self._compute_lnlike(self._check_dimensions(y)[self.inds])
        return ll if np.isfinite(ll) else -np.inf

    def grad_lnlikelihood(self, y):
        if not self.computed:
            raise RuntimeError("You need to compute the model first")

        r = self._check_dimensions(y)[self.inds]

        # Pre-compute some factors.
        alpha = cho_solve(self._factor, r)
        aaT = alpha[:, None] * alpha[None, :]
        Kg = self.kernel.grad(self._x[:, None], self._x[None, :])

        # FIXME: make faster.

        # Loop over dimensions and compute the gradient in each one.
        g = np.empty(len(Kg))
        for i, k in enumerate(Kg):
            d = np.diag(np.dot(aaT, k) - cho_solve(self._factor, k))
            g[i] = 0.5 * np.sum(d)

        return g

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
        if not self.computed:
            raise RuntimeError("You need to compute the model first")

        r = self._check_dimensions(y)[self.inds]
        xs, i = self._parse_samples(t, False)
        alpha = cho_solve(self._factor, r)

        # Compute the predictive mean.
        Kxs = self.kernel(self._x[None, :], xs[:, None])
        mu = np.dot(Kxs, alpha)

        # Compute the predictive covariance.
        cov = self.kernel(xs[:, None], xs[None, :])
        cov -= np.dot(Kxs, cho_solve(self._factor, Kxs.T))

        return mu, cov

    def sample_conditional(self, y, t, size=1):
        """
        Draw samples from the predictive conditional distribution.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param size: (optional)
            The number of samples to draw.

        :returns samples: ``(N, ntest)``
            A list of predictions at coordinates given by ``t``.

        """
        mu, cov = self.predict(y, t)
        return multivariate_gaussian_samples(cov, size, mean=mu)

    def sample(self, t, size=1):
        """
        Draw samples from the prior distribution.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the model should be sampled.

        :param N: (optional)
            The number of samples to draw.

        :returns samples: ``(N, ntest)``
            A list of predictions at coordinates given by ``t``.

        """
        cov = self.get_matrix(t)
        return multivariate_gaussian_samples(cov, size)

    def get_matrix(self, t):
        r, _ = self._parse_samples(t, False)
        return self.kernel(r[:, None], r[None, :])
