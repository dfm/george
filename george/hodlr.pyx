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
    return HODLRSolver(kernel_spec, nleaf=nleaf, tol=tol, rebuild=True)


cdef class HODLRSolver:

    cdef object kernel_spec
    cdef kernels.Kernel* kernel
    cdef Solver* solver
    cdef unsigned int nleaf
    cdef double tol

    def __cinit__(self, kernel_spec, unsigned int nleaf=100, double tol=1e-12,
                  rebuild=False):
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
        def __get__(self):
            return self.solver.get_log_determinant()

    property computed:
        def __get__(self):
            return bool(self.solver.get_computed())

    def apply_inverse(self, y0, in_place=False):
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
            return y

        # Do the standard solve.
        cdef np.ndarray[DTYPE_t, ndim=2] alpha = np.empty_like(y, dtype=DTYPE)
        self.solver.apply_inverse(n, nrhs, <double*>y.data, <double*>alpha.data)
        return alpha.reshape(y0.shape)

    def apply_sqrt(self, y0):
        raise NotImplementedError("The sqrt function isn't available in "
                                  "the HODLR solver yet")
