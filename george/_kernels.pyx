# distutils: language = c++
from __future__ import division

cimport cython
cimport kernels

import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def _rebuild(kernel_spec):
    return CythonKernel(kernel_spec)


cdef class CythonKernel:

    cdef kernels.Kernel* kernel
    cdef object kernel_spec

    def __cinit__(self, kernel_spec):
        self.kernel_spec = kernel_spec
        self.kernel = kernels.parse_kernel(kernel_spec)

    def __reduce__(self):
        return _rebuild, (self.kernel_spec, )

    def __dealloc__(self):
        del self.kernel

    @cython.boundscheck(False)
    def value_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x):
        cdef unsigned int n = x.shape[0], ndim = x.shape[1]
        if self.kernel.get_ndim() != ndim:
            raise ValueError("Dimension mismatch")

        # Build the kernel matrix.
        cdef double value
        cdef unsigned int i, j, delta = x.strides[0]
        cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n, n), dtype=DTYPE)
        for i in range(n):
            k[i, i] = self.kernel.value(<double*>(x.data + i*delta),
                                        <double*>(x.data + i*delta))
            for j in range(i + 1, n):
                value = self.kernel.value(<double*>(x.data + i*delta),
                                          <double*>(x.data + j*delta))
                k[i, j] = value
                k[j, i] = value

        return k

    @cython.boundscheck(False)
    def value_general(self, np.ndarray[DTYPE_t, ndim=2] x1,
                      np.ndarray[DTYPE_t, ndim=2] x2):
        # Parse the input kernel spec.
        cdef unsigned int n1 = x1.shape[0], ndim = x1.shape[1], n2 = x2.shape[0]
        if self.kernel.get_ndim() != ndim or x2.shape[1] != ndim:
            raise ValueError("Dimension mismatch")

        # Build the kernel matrix.
        cdef double value
        cdef unsigned int i, j, d1 = x1.strides[0], d2 = x2.strides[0]
        cdef np.ndarray[DTYPE_t, ndim=2] k = np.empty((n1, n2), dtype=DTYPE)
        for i in range(n1):
            for j in range(n2):
                k[i, j] = self.kernel.value(<double*>(x1.data + i*d1),
                                            <double*>(x2.data + j*d2))

        return k

    @cython.boundscheck(False)
    def gradient_symmetric(self, np.ndarray[DTYPE_t, ndim=2] x):
        # Check the input dimensions.
        cdef unsigned int n = x.shape[0], ndim = x.shape[1]
        if self.kernel.get_ndim() != ndim:
            raise ValueError("Dimension mismatch")

        # Get the number of parameters.
        cdef unsigned int size = self.kernel.size()

        # Build the gradient matrix.
        cdef double value
        cdef np.ndarray[DTYPE_t, ndim=3] g = np.empty((n, n, size), dtype=DTYPE)
        cdef unsigned int i, j, k, delta = x.strides[0]
        cdef unsigned int dx = g.strides[0], dy = g.strides[1]
        for i in range(n):
            self.kernel.gradient(<double*>(x.data + i*delta),
                                 <double*>(x.data + i*delta),
                                 <double*>(g.data + i*dx + i*dy))
            for j in range(i + 1, n):
                self.kernel.gradient(<double*>(x.data + i*delta),
                                     <double*>(x.data + j*delta),
                                     <double*>(g.data + i*dx + j*dy))
                for k in range(size):
                    g[j, i, k] = g[i, j, k]

        return g

    @cython.boundscheck(False)
    def gradient_general(self, np.ndarray[DTYPE_t, ndim=2] x1,
                         np.ndarray[DTYPE_t, ndim=2] x2):
        cdef unsigned int n1 = x1.shape[0], ndim = x1.shape[1], n2 = x2.shape[0]
        if self.kernel.get_ndim() != ndim or x2.shape[1] != ndim:
            raise ValueError("Dimension mismatch")

        # Get the number of parameters.
        cdef unsigned int size = self.kernel.size()

        # Build the gradient matrix.
        cdef double value
        cdef np.ndarray[DTYPE_t, ndim=3] g = np.empty((n1, n2, size), dtype=DTYPE)
        cdef unsigned int i, j, k, d1 = x1.strides[0], d2 = x2.strides[0]
        cdef unsigned int dx = g.strides[0], dy = g.strides[1]
        for i in range(n1):
            for j in range(n2):
                self.kernel.gradient(<double*>(x1.data + i*d1),
                                     <double*>(x2.data + j*d2),
                                     <double*>(g.data + i*dx + j*dy))

        return g
