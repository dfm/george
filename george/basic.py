# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BasicSolver"]

import numpy as np
from scipy.linalg import cholesky, cho_solve


class BasicSolver(object):
    """
    This is the most basic solver for use by

    :param kernel:
        A subclass of :class:`Kernel` implementing a kernel function.

    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.computed = False

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
        """

        """
        return cho_solve(self._factor, y, overwrite_b=in_place)

    def apply_sqrt(self, r):
        """

        """
        return np.dot(r, self._factor[0])
