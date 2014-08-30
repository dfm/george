# distutils: language = c++
from __future__ import division

cimport cython
cimport kernels

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

    # Periodic kernels.
    cdef cppclass CosineKernel(Kernel):
        CosineKernel(const unsigned int ndim, const unsigned int dim)

    cdef cppclass ExpSine2Kernel(Kernel):
        ExpSine2Kernel(const unsigned int ndim, const unsigned int dim)


cdef inline kernels.Kernel* parse_kernel(kernel_spec) except *:
    if not hasattr(kernel_spec, "is_kernel"):
        raise TypeError("Invalid kernel")

    # Deal with operators first.
    cdef kernels.Kernel* k1
    cdef kernels.Kernel* k2
    cdef unsigned int n1, n2
    if not kernel_spec.is_kernel:
        k1 = parse_kernel(kernel_spec.k1)
        n1 = k1.get_ndim()
        k2 = parse_kernel(kernel_spec.k2)
        n2 = k2.get_ndim()
        if n1 != n2:
            raise ValueError("Dimension mismatch")

        if kernel_spec.operator_type == 0:
            return new kernels.Sum(n1, k1, k2)
        elif kernel_spec.operator_type == 1:
            return new kernels.Product(n1, k1, k2)
        else:
            raise TypeError("Unknown operator: {0}".format(
                kernel_spec.__class__.__name__))

    # Get the kernel parameters.
    cdef unsigned int ndim = kernel_spec.ndim
    cdef np.ndarray[DTYPE_t, ndim=1] pars = kernel_spec.pars

    cdef kernels.Kernel* kernel
    if kernel_spec.kernel_type == 0:
        kernel = new kernels.ConstantKernel(ndim)

    elif kernel_spec.kernel_type == 1:
        kernel = new kernels.WhiteKernel(ndim)

    elif kernel_spec.kernel_type == 2:
        kernel = new kernels.DotProductKernel(ndim)

    elif kernel_spec.kernel_type == 3:
        if kernel_spec.isotropic:
            kernel = new kernels.ExpKernel[kernels.IsotropicMetric](ndim,
                new kernels.IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new kernels.ExpKernel[kernels.AxisAlignedMetric](ndim,
                new kernels.AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 4:
        if kernel_spec.isotropic:
            kernel = new kernels.ExpSquaredKernel[kernels.IsotropicMetric](ndim,
                new kernels.IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new kernels.ExpSquaredKernel[kernels.AxisAlignedMetric](ndim,
                new kernels.AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 5:
        if kernel_spec.isotropic:
            kernel = new kernels.Matern32Kernel[kernels.IsotropicMetric](ndim,
                new kernels.IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new kernels.Matern32Kernel[kernels.AxisAlignedMetric](ndim,
                new kernels.AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 6:
        if kernel_spec.isotropic:
            kernel = new kernels.Matern52Kernel[kernels.IsotropicMetric](ndim,
                new kernels.IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new kernels.Matern52Kernel[kernels.AxisAlignedMetric](ndim,
                new kernels.AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 7:
        if kernel_spec.isotropic:
            kernel = new kernels.RationalQuadraticKernel[kernels.IsotropicMetric](ndim,
                new kernels.IsotropicMetric(ndim))
        elif kernel_spec.axis_aligned:
            kernel = new kernels.RationalQuadraticKernel[kernels.AxisAlignedMetric](ndim,
                new kernels.AxisAlignedMetric(ndim))
        else:
            raise NotImplementedError("The general metric isn't implemented")

    elif kernel_spec.kernel_type == 8:
        kernel = new kernels.CosineKernel(ndim, kernel_spec.dim)

    elif kernel_spec.kernel_type == 9:
        kernel = new kernels.ExpSine2Kernel(ndim, kernel_spec.dim)

    else:
        raise TypeError("Unknown kernel: {0}".format(
            kernel_spec.__class__.__name__))

    kernel.set_vector(<double*>pars.data)
    return kernel
