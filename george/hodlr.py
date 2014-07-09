# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HODLRGP"]

import time
import numpy as np

from .basic import GP
from ._george import _george


class HODLRGP(GP):
    """
    This is the money object.

    :params pars:
        The hyperparameters of the covariance function. For now, this must be
        a list or vector of length 2: amplitude and standard deviation.

    """

    def __init__(self, kernel, nleaf=100, tol=1e-12):
        self.nleaf = nleaf
        self.tol = tol
        super(HODLRGP, self).__init__(kernel)

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

    def compute(self, x, yerr, sort=True, seed=None):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :params x: ``(nsamples, )``
            The independent coordinates of the data points.

        :params yerr: ``(nsamples, )``
            The uncertainties on the data points at coordinates ``x``.

        """
        if seed is None:
            seed = int(time.time())

        # Parse the input coordinates.
        self._x, self.inds = self._parse_samples(np.array(x), sort)
        self._yerr = self._check_dimensions(yerr)[self.inds]

        return self._gp.compute(self._x, self._yerr, seed)

    def _compute_lnlike(self, r):
        return self._gp.lnlikelihood(r)

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
        return self._gp.predict(self._check_dimensions(y)[self.inds],
                                self._parse_samples(t, False)[0])

    def get_matrix(self, t):
        return self._gp.get_matrix(self._parse_samples(t, False)[0])
