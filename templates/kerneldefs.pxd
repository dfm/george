# distutils: language = c++
from __future__ import division

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef extern from "metrics.h" namespace "george::metrics":
    cdef cppclass Metric:
        pass
    cdef cppclass IsotropicMetric(Metric):
        pass
    cdef cppclass AxisAlignedMetric(Metric):
        pass
    cdef cppclass GeneralMetric(Metric):
        pass


cdef extern from "kernels.h" namespace "george::kernels":

    cdef cppclass Kernel:
        double value (const double* x1, const double* x2)
        void gradient (const double* x1, const double* x2,
                       const unsigned* which, double* grad)
        unsigned int get_ndim () const
        unsigned int size () const
        void set_vector (const double*)
        void set_axis (unsigned i, unsigned value)
        void set_metric_parameter (unsigned i, double value)

    # Operators.
    cdef cppclass Operator(Kernel):
        pass

    cdef cppclass Sum(Operator):
        Sum(Kernel* k1, Kernel* k2)

    cdef cppclass Product(Operator):
        Product(Kernel* k1, Kernel* k2)

    {% for spec in specs %}
    cdef cppclass {{ spec.name }}{%- if spec.stationary -%}[M]{%- endif -%}(Kernel):
        {{ spec.name }} (
            {% for param in spec.params %}
            double {{ param }},
            {%- endfor %}
            {% for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            {%- if spec.stationary -%}
            unsigned blocked,
            double* min_block,
            double* max_block,
            {%- endif %}
            unsigned ndim,
            unsigned naxes
        )

        void set_axis (unsigned i, unsigned value)
        {% if spec.stationary %}
        void set_metric_parameter (unsigned i, double value)
        {% endif %}
    {% endfor %}

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
    cdef Kernel* kernel
    cdef unsigned blocked
    cdef np.ndarray[DTYPE_t, ndim=1] min_block, max_block

    if kernel_spec.stationary:
        ndim = kernel_spec.metric.ndim
        axes = kernel_spec.metric.axes
        blocked = kernel_spec.blocked
        min_block = kernel_spec.min_block
        max_block = kernel_spec.max_block
    else:
        ndim = kernel_spec.ndim
        axes = kernel_spec.axes


    if False:
        pass
    {% for spec in specs %}
    elif kernel_spec.kernel_type == {{ spec.index }}:

        {% if spec.stationary %}
        metric_spec = kernel_spec.metric
        if metric_spec.metric_type == 0:
            kernel = new {{ spec.name }}[IsotropicMetric] (
                {% for param in spec.params %}
                kernel_spec.{{ param }},
                {%- endfor %}
                {% for con in spec.constants %}
                kernel_spec.{{ con.name }},
                {%- endfor %}
                blocked,
                <double*>(min_block.data),
                <double*>(max_block.data),
                ndim,
                len(axes)
            )
        elif metric_spec.metric_type == 1:
            kernel = new {{ spec.name }}[AxisAlignedMetric] (
                {% for param in spec.params %}
                kernel_spec.{{ param }},
                {%- endfor %}
                {% for con in spec.constants %}
                kernel_spec.{{ con.name }},
                {%- endfor %}
                blocked,
                <double*>(min_block.data),
                <double*>(max_block.data),
                ndim,
                len(axes)
            )
        elif metric_spec.metric_type == 2:
            kernel = new {{ spec.name }}[GeneralMetric] (
                {% for param in spec.params %}
                kernel_spec.{{ param }},
                {%- endfor %}
                {% for con in spec.constants %}
                kernel_spec.{{ con.name }},
                {%- endfor %}
                blocked,
                <double*>(min_block.data),
                <double*>(max_block.data),
                ndim,
                len(axes)
            )
        else:
            raise TypeError("Unknown metric type: {0}".format(
                kernel_spec.__class__.__name__))

        for i, p in enumerate(metric_spec.parameters):
            kernel.set_metric_parameter(i, p)
        {% else %}
        kernel = new {{ spec.name }} (
            {% for param in spec.params %}
            kernel_spec.{{ param }},
            {%- endfor %}
            {% for con in spec.constants %}
            kernel_spec.{{ con.name }},
            {%- endfor %}
            ndim,
            len(axes)
        )
        {% endif %}

        for i, a in enumerate(axes):
            kernel.set_axis(i, a)

    {% endfor %}
    else:
        raise TypeError("Unknown kernel: {0}".format(
                        kernel_spec.__class__.__name__))

    return kernel
