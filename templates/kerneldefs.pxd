# distutils: language = c++
from __future__ import division

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef extern from "subspace.h" namespace "george::subspace":

    cdef cppclass Subspace:
        Subspace(const unsigned ndim, const unsigned naxes)
        void set_axis (const unsigned i, const unsigned value)


cdef extern from "metrics.h" namespace "george::metrics":

    cdef cppclass Metric:
        void set_axis (const unsigned i, const unsigned value)
        void set_parameter (const unsigned i, const double value)

    cdef cppclass IsotropicMetric(Metric):
        IsotropicMetric(const unsigned ndim, const unsigned naxes)
        void set_axis (const unsigned i, const unsigned value)
        void set_parameter (const unsigned i, const double value)

    cdef cppclass AxisAlignedMetric(Metric):
        AxisAlignedMetric(const unsigned ndim, const unsigned ndim)
        void set_axis (const unsigned i, const unsigned value)
        void set_parameter (const unsigned i, const double value)

    cdef cppclass GeneralMetric(Metric):
        GeneralMetric(const unsigned ndim, const unsigned int ndim)
        void set_axis (const unsigned i, const unsigned value)
        void set_parameter (const unsigned i, const double value)


cdef extern from "kernels.h" namespace "george::kernels":

    cdef cppclass Kernel:
        double value (const double* x1, const double* x2) const
        void gradient (const double* x1, const double* x2, double* grad) const
        unsigned int get_ndim () const
        unsigned int size () const
        void set_vector (const double*)

    cdef cppclass CustomKernel(Kernel):
        CustomKernel(const unsigned int ndim, const unsigned int size,
                     void* meta,
                     double (*f) (const double* pars, const unsigned int size,
                                  void* meta,
                                  const double* x1, const double* x2,
                                  const unsigned int ndim),
                     void (*g) (const double* pars, const unsigned int size,
                                void* meta,
                                const double* x1, const double* x2,
                                const unsigned int ndim, double* grad))

    # Operators.
    cdef cppclass Operator(Kernel):
        pass

    cdef cppclass Sum(Operator):
        Sum(Kernel* k1, Kernel* k2)

    cdef cppclass Product(Operator):
        Product(Kernel* k1, Kernel* k2)

    {% for spec in specs %}
    cdef cppclass {{ spec.name }}(Kernel):
        {{ spec.name }} (
            {%- if spec.params -%}{% for param in spec.params %}
            double {{ param }},
            {%- endfor %}{% endif %}
            {%- if spec.stationary %}
            Metric* metric
            {%- else %}
            Subspace* subspace
            {%- endif %}
        )
    {% endfor %}


cdef inline double eval_python_kernel (const double* pars,
                                       const unsigned int size, void* meta,
                                       const double* x1, const double* x2,
                                       const unsigned int ndim) except *:
    # Build the arguments for calling the function.
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp>ndim
    x1_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x1)
    x2_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x2)

    shape[0] = <np.npy_intp>size
    pars_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>pars)

    # Call the Python function and return the value.
    cdef object self = <object>meta
    return self.f(x1_arr, x2_arr, pars_arr)


cdef inline void eval_python_kernel_grad (const double* pars,
                                          const unsigned int size,
                                          void* meta,
                                          const double* x1, const double* x2,
                                          const unsigned int ndim, double* grad) except *:
    # Build the arguments for calling the function.
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp>ndim
    x1_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x1)
    x2_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>x2)

    shape[0] = <np.npy_intp>size
    pars_arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>pars)

    # Call the Python function and update the gradient values in place.
    cdef object self = <object>meta
    cdef np.ndarray[DTYPE_t, ndim=1] grad_arr = \
        np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, <void*>grad)
    grad_arr[:] = self.g(x1_arr, x2_arr, pars_arr)


cdef inline Kernel* parse_kernel(kernel_spec) except *:
    if not hasattr(kernel_spec, "is_kernel"):
        raise TypeError("Invalid kernel")

    # Deal with operators first.
    cdef Kernel* k1
    cdef Kernel* k2
    cdef unsigned n1, n2
    if not kernel_spec.is_kernel:
        k1 = parse_kernel(kernel_spec.k1)
        n1 = k1.get_ndim()
        k2 = parse_kernel(kernel_spec.k2)
        n2 = k2.get_ndim()
        if n1 != n2:
            raise ValueError("Dimension mismatch")

        if kernel_spec.operator_type == 0:
            return new Sum(k1, k2)
        elif kernel_spec.operator_type == 1:
            return new Product(k1, k2)
        else:
            raise TypeError("Unknown operator: {0}".format(
                kernel_spec.__class__.__name__))

    cdef unsigned i
    cdef np.ndarray[DTYPE_t, ndim=1] pars = kernel_spec._parameter_vector
    cdef Kernel* kernel
    cdef Metric* metric
    cdef Subspace* subspace

    if kernel_spec.stationary:
        metric_spec = kernel_spec.metric
        if metric_spec.metric_type == 0:
            metric = new IsotropicMetric(metric_spec.ndim,
                                         len(metric_spec.axes))
        elif metric_spec.metric_type == 1:
            metric = new AxisAlignedMetric(metric_spec.ndim,
                                           len(metric_spec.axes))
        elif metric_spec.metric_type == 2:
            metric = new GeneralMetric(metric_spec.ndim,
                                       len(metric_spec.axes))
        else:
            raise TypeError("Unknown metric type: {0}".format(
                kernel_spec.__class__.__name__))

        for i, (a, p) in enumerate(zip(metric_spec.axes,
                                       metric_spec.parameters)):
            metric.set_axis(i, a)
            metric.set_parameter(i, p)

    else:
        subspace = new Subspace(kernel_spec.ndim, len(kernel_spec.axes))
        for i, a in enumerate(kernel_spec.axes):
            subspace.set_axis(i, a)

    if False:
        pass
    {% for spec in specs %}
    elif kernel_spec.kernel_type == {{ spec.index }}:
        kernel = new {{ spec.name }} (
            {%- if spec.params -%}{% for param in spec.params %}
            kernel_spec.{{ param }},
            {%- endfor %}{% endif %}
            {%- if spec.stationary %}
            metric
            {%- else %}
            subspace
            {%- endif %}
        )
    {% endfor %}
    else:
        raise TypeError("Unknown kernel: {0}".format(
                        kernel_spec.__class__.__name__))

    # if kernel_spec.kernel_type == -2:
    #     kernel = new CustomKernel(ndim, kernel_spec.size, <void*>kernel_spec,
    #                               &eval_python_kernel, &eval_python_kernel_grad)

    # kernel.set_vector(<double*>pars.data)
    return kernel
