# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BasicSolver"]

import numpy as np
from scipy.linalg import cholesky, cho_solve


class BasicSolver(object):
    """
    This is the most basic solver built using :func:`scipy.linalg.cholesky`.

    :param kernel:
        A subclass of :class:`Kernel` implementing a kernel function.

    """

    def __init__(self, kernel):
        self.kernel = kernel
        self._computed = False
        self._log_det = None

    @property
    def computed(self):
        """
        A flag indicating whether or not the covariance matrix was computed
        and factorized (using the :func:`compute` method).

        """
        return self._computed

    @computed.setter
    def computed(self, v):
        self._computed = v

    @property
    def log_determinant(self):
        """
        The log-determinant of the covariance matrix. This will only be
        non-``None`` after calling the :func:`compute` method.

        """
        return self._log_det

    @log_determinant.setter
    def log_determinant(self, v):
        self._log_det = v

    def compute(self, x, yerr):
        """
        Compute and factorize the covariance matrix.

        :param x: ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param yerr: (optional) ``(nsamples,)`` or scalar
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        """
        # Compute the kernel matrix.
        K = self.kernel.value(x)
        K[np.diag_indices_from(K)] += yerr ** 2

        # Factor the matrix and compute the log-determinant.
        self._factor = (cholesky(K, overwrite_a=True, lower=False), False)
        self.log_determinant = 2 * np.sum(np.log(np.diag(self._factor[0])))
        self.computed = True

    def apply_inverse(self, y, in_place=False):
        r"""
        Apply the inverse of the covariance matrix to the input by solving

        .. math::

            C\,x = b

        :param y: ``(nsamples,)`` or ``(nsamples, nrhs)``
            The vector or matrix :math:`b`.

        :param in_place: (optional)
            Should the data in ``y`` be overwritten with the result :math:`x`?

        """
        return cho_solve(self._factor, y, overwrite_b=in_place)

    def apply_sqrt(self, r):
        """
        Apply the Cholesky square root of the covariance matrix to the input
        vector or matrix.

        :param r: ``(nsamples,)`` or ``(nsamples, nrhs)``
            The input vector or matrix.

        """
        return np.dot(r, self._factor[0])
