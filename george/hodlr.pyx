# distutils: language = c++
from __future__ import division

cimport cython
cimport kernels

import time
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "solver.h" namespace "george":

    cdef cppclass Solver:
        Solver(kernels.Kernel*, unsigned int, double)
        int compute (const unsigned int, const double*, const double*, unsigned int)
        int get_computed () const
        double get_log_determinant () const
        void apply_inverse (const unsigned int, const unsigned int, double*, double*)


def _rebuild(kernel_spec, nleaf, tol):
    return HODLRSolver(kernel_spec, nleaf=nleaf, tol=tol)


cdef class HODLRSolver:
    """
    A solver using Sivaram Amambikasaran's `HODLR library
    <https://github.com/sivaramambikasaran/HODLR>`_ that implements a
    :math:`\mathcal{O}(N\,\log^2 N)` direct solver for dense matrices as
    described `here <http://arxiv.org/abs/1403.6015>`_.

    :param kernel:
        A subclass of :class:`Kernel` implementing a kernel function.

    :param nleaf: (optional)
        The size of the smallest matrix blocks. When the solver reaches this
        level in the tree, it directly solves these systems using Eigen's
        Cholesky implementation. (default: ``100``)

    :param tol: (optional)
        A tuning parameter used when factorizing the matrix. The conversion
        between this parameter and the precision of the results is problem
        specific but if you need more precision, try deceasing this number
        (at the cost of a longer runtime). (default: ``1e-12``)

    """

    cdef object kernel_spec
    cdef kernels.Kernel* kernel
    cdef Solver* solver
    cdef unsigned int nleaf
    cdef double tol

    def __cinit__(self, kernel_spec, unsigned int nleaf=100, double tol=1e-12):
        self.kernel_spec = kernel_spec
        self.kernel = kernels.parse_kernel(kernel_spec)
        self.solver = new Solver(self.kernel, nleaf, tol)
        self.nleaf = nleaf
        self.tol = tol

    def __reduce__(self):
        return _rebuild, (self.kernel_spec, self.nleaf, self.tol)

    def __dealloc__(self):
        del self.solver
        del self.kernel

    def compute(self, np.ndarray[DTYPE_t, ndim=2] x,
                np.ndarray[DTYPE_t, ndim=1] yerr, seed=None):
        """
        compute(x, yerr, seed=None)
        Compute and factorize the covariance matrix.

        :param x: ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param yerr: (optional) ``(nsamples,)`` or scalar
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        :param seed: (optional)
            There is a stochastic component in the HODLR factorization step.
            Use this parameter (it should be an integer) to seed this random
            step and ensure deterministic results. Normally the randomization
            shouldn't make a big difference but as the matrix becomes poorly
            conditioned, it will have a larger effect.

        """
        # Check the input dimensions.
        cdef unsigned int n = x.shape[0]
        cdef unsigned int ndim = x.shape[1]
        if yerr.shape[0] != n or ndim != self.kernel.get_ndim():
            raise ValueError("Dimension mismatch")

        # Seed with the time if no seed is provided.
        if seed is None:
            seed = time.time()

        # Compute the matrix.
        cdef int info
        info = self.solver.compute(n, <double*>x.data, <double*>yerr.data, seed)
        if info != 0:
            raise np.linalg.LinAlgError(info)

    property log_determinant:
        """
        The log-determinant of the covariance matrix. This will only be
        non-``None`` after calling the :func:`compute` method.

        """
        def __get__(self):
            return self.solver.get_log_determinant()

    property computed:
        """
        A flag indicating whether or not the covariance matrix was computed
        and factorized (using the :func:`compute` method).

        """
        def __get__(self):
            return bool(self.solver.get_computed())

    def apply_inverse(self, y0, in_place=False):
        """
        apply_inverse(y, in_place=False)
        Apply the inverse of the covariance matrix to the input by solving

        .. math::

            C\,x = b

        :param y: ``(nsamples,)`` or ``(nsamples, nrhs)``
            The vector or matrix :math:`b`.

        :param in_place: (optional)
            Should the data in ``y`` be overwritten with the result :math:`x`?

        """
        # Coerce the input array into the correct format.
        cdef np.ndarray[DTYPE_t, ndim=2] y
        if len(y0.shape) == 1:
            y = np.atleast_2d(y0).T
        else:
            y = y0

        # Get the problem dimensions.
        cdef unsigned int n = y.shape[0], nrhs = y.shape[1]

        # Do an in-place solve if requested.
        if in_place:
            self.solver.apply_inverse(n, nrhs, <double*>y.data, <double*>y.data)
            return y.reshape(y0.shape)

        # Do the standard solve.
        cdef np.ndarray[DTYPE_t, ndim=2] alpha = np.empty_like(y, dtype=DTYPE)
        self.solver.apply_inverse(n, nrhs, <double*>y.data, <double*>alpha.data)
        return alpha.reshape(y0.shape)

    def apply_sqrt(self, y0):
        """
        apply_sqrt(r)
        This method is not implemented by this solver yet.

        """
        raise NotImplementedError("The sqrt function isn't available in "
                                  "the HODLR solver yet")
