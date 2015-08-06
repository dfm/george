# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Kernel", "Sum", "Product",
    {%- for spec in specs %}
    "{{ spec.name }}",
    {%- endfor %}
]

import numpy as np
from functools import partial

from .utils import numerical_gradient
from .metrics import Metric, Subspace
from .cython_kernel import CythonKernel


class Kernel(object):
    """
    The abstract kernel type. Every kernel implemented in George should be
    a subclass of this object.

    :param pars:
        The hyper-parameters of the kernel.

    :param ndim: (optional)
        The number of input dimensions of the kernel. (default: ``1``)

    """

    is_kernel = True
    is_radial = False
    kernel_type = -1

    # This function deals with weird behavior when performing arithmetic
    # operations with numpy scalars.
    def __array_wrap__(self, array, context=None):
        if context is None:
            raise TypeError("Invalid operation")
        ufunc, args, _ = context
        if ufunc.__name__ == "multiply":
            return float(args[0]) * args[1]
        elif ufunc.__name__ == "add":
            return float(args[0]) + args[1]
        raise TypeError("Invalid operation")
    __array_priority__ = np.inf

    def __init__(self, *pars, **kwargs):
        self.ndim = kwargs.get("ndim", 1)
        self.pars = np.array(pars)
        self.dirty = True
        self._kernel = None

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict["_kernel"] = None
        return odict

    @property
    def kernel(self):
        if self.dirty or self._kernel is None:
            self._kernel = CythonKernel(self)
            self.dirty = False
        return self._kernel

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map("{0}".format,
                                               self.pars) +
                                           ["ndim={0}".format(self.ndim)]))

    def lnprior(self):
        return 0.0

    @property
    def vector(self):
        return np.log(self.pars)

    @vector.setter
    def vector(self, v):
        self.pars = np.exp(v)

    @property
    def pars(self):
        return self._pars

    @pars.setter
    def pars(self, v):
        self._pars = np.array(v, dtype=np.float64, order="C")
        self.dirty = True

    def __getitem__(self, i):
        return self.vector[i]

    def __setitem__(self, i, v):
        vec = self.vector
        vec[i] = v
        self.vector = vec

    def __len__(self):
        return len(self.pars)

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(value=float(b), ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(value=float(b), ndim=self.ndim),
                           self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)

    def value(self, x1, x2=None):
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            return self.kernel.value_symmetric(x1)
        x2 = np.ascontiguousarray(x2, dtype=np.float64)
        return self.kernel.value_general(x1, x2)

    def gradient(self, x1, x2=None):
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            g = self.kernel.gradient_symmetric(x1)
        else:
            x2 = np.ascontiguousarray(x2, dtype=np.float64)
            g = self.kernel.gradient_general(x1, x2)
        return g * self.vector_gradient[None, None, :]

    @property
    def vector_gradient(self):
        return self.pars


class _operator(Kernel):
    is_kernel = False
    operator_type = -1

    def __init__(self, k1, k2):
        if k1.ndim != k2.ndim:
            raise ValueError("Dimension mismatch")
        self.k1 = k1
        self.k2 = k2
        self.ndim = k1.ndim
        self._dirty = True
        self._kernel = None

    def lnprior(self):
        return self.k1.lnprior() + self.k2.lnprior()

    @property
    def dirty(self):
        return self._dirty or self.k1.dirty or self.k2.dirty

    @dirty.setter
    def dirty(self, v):
        self._dirty = v
        self.k1.dirty = False
        self.k2.dirty = False

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)

    @pars.setter
    def pars(self, v):
        self._dirty = True
        i = len(self.k1)
        self.k1.pars = v[:i]
        self.k2.pars = v[i:]

    @property
    def vector(self):
        return np.append(self.k1.vector, self.k2.vector)

    @vector.setter
    def vector(self, v):
        self._dirty = True
        i = len(self.k1)
        self.k1.vector = v[:i]
        self.k2.vector = v[i:]


class Sum(_operator):
    is_kernel = False
    operator_type = 0

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)


class Product(_operator):
    is_kernel = False
    operator_type = 1

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)

{% for spec in specs %}
class {{ spec.name }} (Kernel):
    """
    {{ spec.doc | indent(4) }}

    """

    kernel_type = {{ spec.index }}
    stationary = {{ spec.stationary }}
    parameter_names = [{% for p in spec.params -%}"{{ p }}", {% endfor %}]

    def __init__(self,
                 {% for p in spec.params %}{{ p }}=None,
                 {% endfor -%}
                 {% if spec.stationary -%}
                 metric=1.0,
                 lower=True,
                 {% endif -%}
                 ndim=1,
                 axes=None):
        {% for p in spec.params -%}
        if {{ p }} is None:
            raise ValueError("missing required parameter '{{ p }}'")
        self.{{ p }} = {{ p }}
        {% endfor -%}

        {% if spec.stationary -%}
        self.metric = Metric(metric, ndim=ndim, axes=axes, lower=lower)
        {%- else -%}
        self.subspace = Subspace(ndim, axes=axes)
        {%- endif %}

{% endfor %}


class RadialKernel(Kernel):
    r"""
    This kernel (and more importantly its subclasses) computes the distance
    between two samples in an arbitrary metric and applies a radial function
    to this distance.

    :param metric:
        The specification of the metric. This can be a ``float``, in which
        case the metric is considered isotropic with the variance in each
        dimension given by the value of ``metric``. Alternatively, ``metric``
        can be a list of variances for each dimension. In this case, it should
        have length ``ndim``. The fully general (not axis-aligned) metric
        hasn't been implemented yet but it's on the to do list!

    :param dim: (optional)
        If provided, this will apply the kernel in only the specified
        dimension.

    """
    is_radial = True

    def __init__(self, metric, ndim=1, dim=-1, extra=[]):
        self.isotropic = False
        self.axis_aligned = False
        try:
            float(metric)

        except TypeError:
            metric = np.atleast_1d(metric)
            if len(metric) == ndim:
                # The metric is axis aligned.
                self.axis_aligned = True

            else:
                raise NotImplementedError("The general metric isn't "
                                          "implemented")

        else:
            # If we get here then the kernel is isotropic.
            self.isotropic = True

        if dim >= 0:
            assert self.isotropic, "A 1-D kernel should also be isotropic"
        self.dim = dim

        super(RadialKernel, self).__init__(*(np.append(extra, metric)),
                                           ndim=ndim)


class PythonKernel(Kernel):
    r"""
    A custom kernel evaluated in Python. The gradient is optionally evaluated
    numerically. For big problems, this type of kernel will probably be
    unbearably slow because each evaluation is done point-wise. Unfortunately,
    this is the only way to implement custom kernels without re-compiling
    George. Hopefully we can solve this in the future!

    :param f:
        A callable that evaluates the kernel function given arguments
        ``(x1, x2, p)`` where ``x1`` and ``x2`` are numpy array defining the
        coordinates of the samples and ``p`` is the numpy array giving the
        current settings of the parameters.

    :param g: (optional)
        A function with the same calling parameters as ``f`` but it should
        return the numpy array with the gradient of the kernel function. If
        this function isn't given then the gradient is evaluated using
        centered finite difference.

    :param pars: (optional)
        The initial list of parameter values. If this isn't provided then the
        kernel is assumed to have no parameters.

    :param dx: (optional)
        The step size used for the gradient computation when using finite
        difference.

    """

    kernel_type = -2

    def __init__(self, f, g=None, pars=(), dx=1.234e-6, ndim=1):
        super(PythonKernel, self).__init__(*pars, ndim=ndim)
        self.size = len(self.pars)
        self.f = f
        self.g = self._wrap_grad(f, g, dx=dx)

    def _wrap_grad(self, f, g, dx=1.234e-6):
        if g is not None:
            grad = g
        else:
            def grad(x1, x2, p):
                g = numerical_gradient(partial(f, x1, x2), p, dx=dx)
                return g
        return grad
