# distutils: language = c++
from __future__ import division

import numpy as np
cimport numpy as np

from scipy.optimize import check_grad

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_I = np.uint32
ctypedef np.uint32_t DTYPE_I_t


cdef extern from "metrics.h" namespace "george::metrics":

    cdef cppclass Metric:
        unsigned int size ()
        void set_parameter (const unsigned int i, const double value)
        double get_parameter (const unsigned int i)
        void set_axis (const unsigned int i, const unsigned int value)
        unsigned int get_axis (const unsigned int i) const

    cdef cppclass IsotropicMetric(Metric):
        IsotropicMetric(const unsigned int naxes)
        double value (const double* x1, const double* x2)
        double gradient (const double* x1, const double* x2, double* grad)

    cdef cppclass AxisAlignedMetric(Metric):
        AxisAlignedMetric(const unsigned int naxes)
        double value (const double* x1, const double* x2)
        double gradient (const double* x1, const double* x2, double* grad)

    cdef cppclass GeneralMetric(Metric):
        GeneralMetric(const unsigned int naxes)
        double value (const double* x1, const double* x2)
        double gradient (const double* x1, const double* x2, double* grad)


def test_metrics():
    cdef GeneralMetric* m = new GeneralMetric(2)
    m.set_axis(0, 0)
    m.set_axis(1, 1)
    m.set_parameter(0, 1.0)
    m.set_parameter(1, 100.0)
    m.set_parameter(2, 0.5)

    cdef int ndim = 5
    cdef double eps = 1.26e-8
    cdef np.ndarray[DTYPE_t] x1 = np.random.randn(ndim)
    cdef np.ndarray[DTYPE_t] x2 = 0.5+np.random.randn(ndim)
    cdef np.ndarray[DTYPE_t] grad = np.zeros(3, dtype=DTYPE)

    cdef int i
    cdef double vp, vm
    for i in range(3):
        m.gradient(<double*>x1.data, <double*>x2.data, <double*>grad.data)
        m.set_parameter(i, m.get_parameter(i)+eps)
        vp = m.value(<double*>x1.data, <double*>x2.data)
        m.set_parameter(i, m.get_parameter(i)-2*eps)
        vm = m.value(<double*>x1.data, <double*>x2.data)
        m.set_parameter(i, m.get_parameter(i)+eps)
        print((vp - vm)/(2*eps))
    print(grad)
