# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Kernel", "Sum", "Product",
    {%- for spec in specs %}
    "{{ spec.name }}",
    {%- endfor %}
]

import numpy as np

from .modeling import Model, ModelSet
from .metrics import Metric, Subspace
from .kernel_interface import KernelInterface


class Kernel(ModelSet):
    """
    The abstract kernel type. Every kernel implemented in George should be
    a subclass of this object.

    """

    is_kernel = True
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

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict["_kernel"] = None
        return odict

    # We must overload the ModelSet attribute getter to pass the requests
    # to the "BaseKernel"
    def __getattr__(self, name):
        if "models" in self.__dict__:
            if name in self.models:
                return self.models[name]
            if None in self.models:
                return getattr(self.models[None], name)
        raise AttributeError(name)

    @property
    def kernel(self):
        return KernelInterface(self)

    def __repr__(self):
        kernel = self.models[None]
        params = ["{0}={1}".format(k, getattr(kernel, k))
                  for k in kernel.parameter_names]
        if self.stationary:
            params += ["metric={0}".format(repr(self.metric)),
                       "block={0}".format(repr(self.block))]
        else:
            params += ["ndim={0}".format(self.ndim),
                       "axes={0}".format(repr(self.axes))]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(params))

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(log_constant=np.log(float(b)/self.ndim),
                                      ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            log_constant = np.log(float(b)/self.ndim)
            return Product(ConstantKernel(log_constant=log_constant,
                                          ndim=self.ndim), self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)

    def get_value(self, x1, x2=None, diag=False):
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            if diag:
                return self.kernel.value_diagonal(x1, x1)
            else:
                return self.kernel.value_symmetric(x1)
        x2 = np.ascontiguousarray(x2, dtype=np.float64)
        if diag:
            return self.kernel.value_diagonal(x1, x2)
        else:
            return self.kernel.value_general(x1, x2)

    def get_gradient(self, x1, x2=None, include_frozen=False):
        mask = (
            np.ones(self.full_size, dtype=bool)
            if include_frozen else self.unfrozen_mask
        )
        which = mask.astype(np.uint32)
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            g = self.kernel.gradient_symmetric(which, x1)
        else:
            x2 = np.ascontiguousarray(x2, dtype=np.float64)
            g = self.kernel.gradient_general(which, x1, x2)
        return g[:, :, mask]

    def get_x1_gradient(self, x1, x2=None):
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            x2 = x1
        else:
            x2 = np.ascontiguousarray(x2, dtype=np.float64)
        return self.kernel.x1_gradient_general(x1, x2)

    def get_x2_gradient(self, x1, x2=None):
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            x2 = x1
        else:
            x2 = np.ascontiguousarray(x2, dtype=np.float64)
        return self.kernel.x2_gradient_general(x1, x2)

    def test_gradient(self, x1, x2=None, eps=1.32e-6, **kwargs):
        vector = self.get_parameter_vector()
        g0 = self.get_gradient(x1, x2=x2)

        for i, v in enumerate(vector):
            vector[i] = v + eps
            self.set_parameter_vector(vector)
            kp = self.get_value(x1, x2=x2)

            vector[i] = v - eps
            self.set_parameter_vector(vector)
            km = self.get_value(x1, x2=x2)

            vector[i] = v
            self.set_parameter_vector(vector)

            grad = 0.5 * (kp - km) / eps
            assert np.allclose(g0[:, :, i], grad, **kwargs), \
                "incorrect gradient for parameter '{0}' ({1})" \
                .format(self.get_parameter_names()[i], i)

    def test_x1_gradient(self, x1, x2=None, eps=1.32e-6, **kwargs):
        kwargs["atol"] = kwargs.get("atol", 0.5 * eps)
        g0 = self.get_x1_gradient(x1, x2=x2)
        if x2 is None:
            x2 = np.array(x1)
        for i in range(len(x1)):
            for k in range(self.ndim):
                x1[i, k] += eps
                kp = self.get_value(x1, x2=x2)

                x1[i, k] -= 2*eps
                km = self.get_value(x1, x2=x2)

                x1[i, k] += eps

                grad = 0.5 * (kp - km) / eps
                assert np.allclose(g0[i, :, k], grad[i], **kwargs)

    def test_x2_gradient(self, x1, x2=None, eps=1.32e-6, **kwargs):
        kwargs["atol"] = kwargs.get("atol", 0.5 * eps)
        g0 = self.get_x2_gradient(x1, x2=x2)
        if x2 is None:
            x2 = np.array(x1)
        for i in range(len(x2)):
            for k in range(self.ndim):
                x2[i, k] += eps
                kp = self.get_value(x1, x2=x2)

                x2[i, k] -= 2*eps
                km = self.get_value(x1, x2=x2)

                x2[i, k] += eps

                grad = 0.5 * (kp - km) / eps
                assert np.allclose(g0[:, i, k], grad[:, i], **kwargs)


class _operator(Kernel):
    is_kernel = False
    kernel_type = -1
    operator_type = -1

    def __init__(self, k1, k2):
        if k1.ndim != k2.ndim:
            raise ValueError("Dimension mismatch")
        self.ndim = k1.ndim
        self._dirty = True
        super(_operator, self).__init__([("k1", k1), ("k2", k2)])

    @property
    def k1(self):
        return self.models["k1"]

    @property
    def k2(self):
        return self.models["k2"]

    @property
    def dirty(self):
        return self._dirty or self.k1.dirty or self.k2.dirty

    @dirty.setter
    def dirty(self, v):
        self._dirty = v
        self.k1.dirty = False
        self.k2.dirty = False


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
class Base{{ spec.name }} (Model):
    parameter_names = ({% for p in spec.params -%}"{{ p }}", {% endfor %})


class {{ spec.name }} (Kernel):
    r"""
    {{ spec.doc | indent(4) }}

    """

    kernel_type = {{ spec.index }}
    stationary = {{ spec.stationary }}

    def __init__(self,
                 {% for p in spec.params %}{{ p }}=None,
                 {% endfor -%}
                 {% for con in spec.constants %}{{ con.name }}=None,
                 {% endfor -%}
                 {% if spec.stationary -%}
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 {% endif -%}
                 bounds=None,
                 ndim=1,
                 axes=None):
        {% for con in spec.constants %}
        if {{ con.name }} is None:
            raise ValueError("missing required parameter '{{ con.name }}'")
        self.{{ con.name }} = {{ con.name }}
        {% endfor %}
        {% if spec.stationary -%}
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block
        {%- else -%}
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes
        {%- endif %}

        kwargs = dict({% for p in spec.params -%}{{ p }}={{ p }}, {% endfor -%})
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = Base{{ spec.name }}(**kwargs)
        super({{ spec.name }}, self).__init__([
            (None, base), {% if spec.stationary -%}("metric", metric){%- endif %}
        ])

        # Common setup.
        self.dirty = True
    {% if spec.stationary %}
    @property
    def block(self):
        if not self.blocked:
            return None
        return list(zip(self.min_block, self.max_block))

    @block.setter
    def block(self, block):
        if block is None:
            self.blocked = False
            self.min_block = -np.inf + np.zeros(len(self.axes))
            self.max_block = np.inf + np.zeros(len(self.axes))
            return

        block = np.atleast_2d(block)
        if block.shape != (len(self.axes), 2):
            raise ValueError("dimension mismatch in block specification")
        self.blocked = True
        self.min_block, self.max_block = map(np.array, zip(*block))
    {% endif %}
{% endfor %}
