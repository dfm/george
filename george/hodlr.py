# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HODLRGP"]

import time

from .basic import GP
from ._george import _george


class HODLRGP(GP):
    """
    A solver built on top of Sivaram Amambikasaran's `HODLR library
    <https://github.com/sivaramambikasaran/HODLR>`_ for linear algebra. The
    HODLR library includes an :math:`\mathcal{O}(N\,\log^2 N)` direct solver
    for dense matrices as described `here <http://arxiv.org/abs/1403.6015>`_.

    :param kernel:
        An instance of a subclass of :class:`kernels.Kernel`.

    :param mean: (optional)
        A description of the mean function; can be a callable or a scalar. If
        scalar, the mean is assumed constant. Otherwise, the function will be
        called with the array of independent coordinates as the only argument.
        (default: ``0.0``)

    :param nleaf: (optional)
        A tuning parameter for the HODLR algorithm. This parameter sets the
        size of the smallest leaf in the tree. (default: ``100``)

    :param tol: (optional)
        A tuning parameter for the HODLR algorithm. This parameter sets the
        low-rank tolerance of the pivoting algorithm. (default: ``1e-12``)

    """

    def __init__(self, kernel, nleaf=100, tol=1e-12, mean=None):
        self.nleaf = nleaf
        self.tol = tol
        self._gp = None
        super(HODLRGP, self).__init__(kernel, mean=mean)

    @property
    def computed(self):
        """
        Has the processes been computed since the last update of the kernel?

        """
        return (self._gp is not None and self._gp.computed()
                and not self.kernel.dirty)

    @property
    def gp(self):
        if self._gp is None or not self.computed:
            self._gp = _george(self.kernel, self.nleaf, self.tol)
        return self._gp

    def compute(self, x, yerr, sort=True, seed=None):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :param x: ``(nsamples,)`` or ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param yerr: ``(nsamples,)``
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        :param sort: (optional)
            Should the samples be sorted before computing the covariance
            matrix? This can lead to more numerically stable results and with
            some linear algebra libraries this can more computationally
            efficient. Either way, this flag is passed directly to
            :func:`parse_samples`.

        """
        if seed is None:
            seed = int(time.time())

        # Parse the input coordinates.
        self._x, self.inds = self.parse_samples(x, sort)
        self._yerr = self._check_dimensions(yerr)[self.inds]

        return self.gp.compute(self._x, self._yerr, seed)

    def _compute_lnlike(self, r):
        return self.gp.lnlikelihood(r)

    def grad_lnlikelihood(self, y):
        raise NotImplementedError("Gradients have not been implemented in the "
                                  "HODLR solver yet.")

    def predict(self, y, t):
        """
        Compute the conditional predictive distribution of the model.

        :param y: ``(nsamples,)``
            The observations to condition the model on.

        :param t: ``(ntest,)`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        Returns a tuple ``(mu, cov)`` where

        * **mu** ``(ntest,)`` is the mean of the predictive distribution, and
        * **cov** ``(ntest, ntest)`` is the predictive covariance.

        """
        r = self._check_dimensions(y)[self.inds] - self.mean(self._x)
        return self.gp.predict(r, self.parse_samples(t, False)[0])

    def get_matrix(self, t):
        """
        Get the covariance matrix at a given set of independent coordinates.

        :param t: ``(nsamples,)`` or ``(nsamples, ndim)``
            The list of samples.

        """
        return self.gp.get_matrix(self.parse_samples(t, False)[0])
