# distutils: language = c++
from __future__ import division

cimport cython

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "metrics.h" namespace "george::metrics":

    cdef cppclass Metric:
        pass

    cdef cppclass IsotropicMetric(Metric):
        IsotropicMetric(const unsigned int ndim)

    cdef cppclass AxisAlignedMetric(Metric):
        AxisAlignedMetric(const unsigned int ndim)

cdef extern from "kernels.h" namespace "george::kernels":

    cdef cppclass Kernel:
        double value (const double* x1, const double* x2) const
        void gradient (const double* x1, const double* x2, double* grad) const
        unsigned int get_ndim () const
        unsigned int size () const
        void set_vector (const double*)

    # Operators.
    cdef cppclass Operator(Kernel):
        pass

    cdef cppclass Sum(Operator):
        Sum(const unsigned int ndim, Kernel* k1, Kernel* k2)

    cdef cppclass Product(Operator):
        Product(const unsigned int ndim, Kernel* k1, Kernel* k2)

    # Basic kernels.
    cdef cppclass ConstantKernel(Kernel):
        ConstantKernel(const unsigned int ndim)

    cdef cppclass WhiteKernel(Kernel):
        WhiteKernel(const unsigned int ndim)

    cdef cppclass DotProductKernel(Kernel):
        DotProductKernel(const unsigned int ndim)

    # Radial kernels.
    cdef cppclass ExpKernel[M](Kernel):
        ExpKernel(const unsigned int ndim, M* metric)

    cdef cppclass ExpSquaredKernel[M](Kernel):
        ExpSquaredKernel(const unsigned int ndim, M* metric)

    cdef cppclass Matern32Kernel[M](Kernel):
        Matern32Kernel(const unsigned int ndim, M* metric)

    cdef cppclass Matern52Kernel[M](Kernel):
        Matern52Kernel(const unsigned int ndim, M* metric)

    cdef cppclass RationalQuadraticKernel[M](Kernel):
        RationalQuadraticKernel(const unsigned int ndim, M* metric)


cdef class CythonKernel:

    cdef Kernel* kernel

    def __cinit__(self, kernel_spec):
        self.kernel = build_kernel(kernel_spec)

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


cdef Kernel* build_kernel(kernel_spec) except *:
    if not hasattr(kernel_spec, "is_kernel"):
        raise TypeError("Invalid kernel")

    # Deal with operators first.
    cdef Kernel* k1
    cdef Kernel* k2
    cdef unsigned int n1, n2
    if not kernel_spec.is_kernel:
        k1 = build_kernel(kernel_spec.k1)
        n1 = k1.get_ndim()
        k2 = build_kernel(kernel_spec.k2)
        n2 = k2.get_ndim()
        if n1 != n2:
            raise ValueError("Dimension mismatch")

        if kernel_spec.operator_type == 0:
            return new Sum(n1, k1, k2)
        elif kernel_spec.operator_type == 1:
            return new Product(n1, k1, k2)
        else:
            raise TypeError("Unknown operator: {0}".format(
                kernel_spec.__class__.__name__))

    # Get the kernel parameters.
    cdef unsigned int ndim = kernel_spec.ndim
    cdef np.ndarray[DTYPE_t, ndim=1] pars = kernel_spec.pars

    # Parse the metric for radial kernels.
    cdef Metric* metric
    cdef Kernel* kernel
    if kernel_spec.is_radial:
        if kernel_spec.isotropic:
            metric = new IsotropicMetric(ndim)
        elif kernel_spec.axis_aligned:
            metric = new AxisAlignedMetric(ndim)
        else:
            raise NotImplementedError("The general metric isn't implemented")

    if kernel_spec.kernel_type == 0:
        kernel = new ConstantKernel(ndim)

    elif kernel_spec.kernel_type == 1:
        kernel = new WhiteKernel(ndim)

    elif kernel_spec.kernel_type == 2:
        kernel = new DotProductKernel(ndim)

    elif kernel_spec.kernel_type == 3:
        if kernel_spec.isotropic:
            kernel = new ExpKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new ExpKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 4:
        if kernel_spec.isotropic:
            kernel = new ExpSquaredKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new ExpSquaredKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 5:
        if kernel_spec.isotropic:
            kernel = new Matern32Kernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Matern32Kernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 6:
        if kernel_spec.isotropic:
            kernel = new Matern52Kernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new Matern52Kernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 7:
        if kernel_spec.isotropic:
            kernel = new RationalQuadraticKernel[IsotropicMetric](ndim,
                new IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new RationalQuadraticKernel[AxisAlignedMetric](ndim,
                new AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    else:
        raise TypeError("Unknown kernel: {0}".format(
            kernel_spec.__class__.__name__))

    kernel.set_vector(<double*>pars.data)
    return kernel
