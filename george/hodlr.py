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

    def _do_compute(self, seed=None):
        if seed is None:
            seed = int(time.time())
        self.gp.compute(self._x, self._yerr, seed)
        self.kernel.dirty = False

    def _compute_lnlike(self, r):
        return self.gp.lnlikelihood(r)

    def grad_lnlikelihood(self, *args, **kwargs):
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
