# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Kernel", "Sum", "Product",
    {%- for spec in specs %}
    "{{ spec.name }}",
    {%- endfor %}
]

import fnmatch
import warnings
import numpy as np
from functools import partial

from .utils import numerical_gradient
from .metrics import Metric, Subspace
from .cython_kernel import CythonKernel
from .compat import imap, basestring, xrange


class Kernel(object):
    """
    The abstract kernel type. Every kernel implemented in George should be
    a subclass of this object.

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
        params = ["{0}={1}".format(k, getattr(self, k))
                  for k in self._parameter_names]
        if self.stationary:
            params += ["metric={0}".format(self.metric),
                       "block={0}".format(self.block)]
        else:
            params += ["ndim={0}".format(self.ndim),
                       "axes={0}".format(self.axes)]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(params))

    def __getattr__(self, k):
        try:
            i = self._parameter_names.index(k)
        except ValueError:
            raise AttributeError("no attribute '{0}'".format(k))
        return self._parameter_vector[i]

    def __setattr__(self, k, v):
        try:
            i = self._parameter_names.index(k)
        except ValueError:
            super(Kernel, self).__setattr__(k, v)
        else:
            self._parameter_vector[i] = v
            self.dirty = True

    def __getitem__(self, k):
        if isinstance(k, basestring):
            return self.get_parameter(k)
        return self.get_vector()[k]

    def __setitem__(self, k, value):
        if isinstance(k, basestring):
            self.set_parameter(k, value)
            return
        v = self.get_vector()
        v[k] = value
        self.set_vector(v)

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(constant=float(b), ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(constant=float(b), ndim=self.ndim),
                                          self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)

    # Kernels must implement these methods to conform with the modeling
    # protocol.
    _parameter_names = []
    _parameter_vector = np.array([])
    _unfrozen = np.array([], dtype=bool)

    @property
    def unfrozen(self):
        uf = self._unfrozen
        if self.stationary:
            uf = np.append(uf, self.metric.unfrozen)
        return uf

    def __len__(self):
        if self.stationary:
            return np.sum(self._unfrozen) + len(self.metric)
        return np.sum(self._unfrozen)

    def get_parameter_names(self, full=False):
        if full:
            n = list(self._parameter_names)
        else:
            n = [n for i, n in enumerate(self._parameter_names)
                 if self._unfrozen[i]]
        if self.stationary:
            n += self.metric.get_parameter_names(full=full)
        return n

    def get_bounds(self):
        bounds = [b for i, b in enumerate(self._parameter_bounds)
                  if self._unfrozen[i]]
        if self.stationary:
            bounds += self.metric.get_bounds()
        return bounds

    def get_vector(self):
        v = self._parameter_vector[self._unfrozen]
        if self.stationary:
            v = np.append(v, self.metric.get_vector())
        return v

    def check_vector(self, vector):
        for i, (a, b) in enumerate(self.get_bounds()):
            v = vector[i]
            if (a is not None and v < a) or (b is not None and b < v):
                return False
        return True

    def set_vector(self, vector):
        if len(self) != len(vector):
            raise ValueError("dimension mismatch")
        n = np.sum(self._unfrozen)
        self._parameter_vector[self._unfrozen] = vector[:n]
        if self.stationary:
            self.metric.set_vector(vector[n:])
        self.dirty = True

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

    def get_gradient(self, x1, x2=None):
        which = self.unfrozen.astype(np.uint32)
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        if x2 is None:
            g = self.kernel.gradient_symmetric(which, x1)
        else:
            x2 = np.ascontiguousarray(x2, dtype=np.float64)
            g = self.kernel.gradient_general(which, x1, x2)
        return g[:, :, self.unfrozen]

    def test_gradient(self, x1, x2=None, eps=1.32e-6, **kwargs):
        vector = self.get_vector()
        g0 = self.get_gradient(x1, x2=x2)

        for i, v in enumerate(vector):
            vector[i] = v + eps
            self.set_vector(vector)
            kp = self.get_value(x1, x2=x2)

            vector[i] = v - eps
            self.set_vector(vector)
            km = self.get_value(x1, x2=x2)

            vector[i] = v
            self.set_vector(vector)

            grad = 0.5 * (kp - km) / eps
            assert np.allclose(g0[:, :, i], grad, **kwargs), \
                "incorrect gradient for parameter '{0}' ({1})" \
                .format(self.get_parameter_names()[i], i)

    def _wildcard_apply(self, meth, pat, *args):
        if len(set("[]*?") & set(pat)):
            names = fnmatch.filter(self.get_parameter_names(full=True), pat)
        else:
            names = [pat]
        elements = []
        for name in names:
            try:
                i = self._parameter_names.index(name)
            except ValueError:
                if self.stationary:
                    elements.append(getattr(self.metric, meth)(name, *args))
            else:
                elements.append(getattr(self, "_" + meth)(i, *args))

        if len(elements) == 0:
            raise ValueError("invalid parameter '{0}'".format(pat))
        return elements

    def _freeze_parameter(self, ind):
        self._unfrozen[ind] = False

    def _thaw_parameter(self, ind):
        self._unfrozen[ind] = True

    def _get_parameter(self, ind):
        return self._parameter_vector[ind]

    def _set_parameter(self, ind, value):
        self._parameter_vector[ind] = value

    def freeze_parameter(self, parameter_name):
        self._wildcard_apply("freeze_parameter", parameter_name)

    def thaw_parameter(self, parameter_name):
        self._wildcard_apply("thaw_parameter", parameter_name)

    def get_parameter(self, parameter_name):
        r = self._wildcard_apply("get_parameter", parameter_name)
        if len(r) == 1:
            r = r[0]
        try:
            return float(r)
        except TypeError:
            return np.array(r, dtype=float)

    def set_parameter(self, parameter_name, value):
        self._wildcard_apply("set_parameter", parameter_name, value)


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

    @property
    def dirty(self):
        return self._dirty or self.k1.dirty or self.k2.dirty

    @dirty.setter
    def dirty(self, v):
        self._dirty = v
        self.k1.dirty = False
        self.k2.dirty = False

    # Modeling protocol:
    @property
    def unfrozen(self):
        return np.append(self.k1.unfrozen, self.k2.unfrozen)

    def __len__(self):
        return len(self.k1) + len(self.k2)

    def get_parameter_names(self, full=False):
        return (
            list(imap("k1:{0}".format, self.k1.get_parameter_names(full=full)))
            +
            list(imap("k2:{0}".format, self.k2.get_parameter_names(full=full)))
        )

    def get_vector(self):
        return np.append(self.k1.get_vector(), self.k2.get_vector())

    def set_vector(self, vector):
        n = len(self.k1)
        self.k1.set_vector(vector[:n])
        self.k2.set_vector(vector[n:])

    def _wildcard_apply(self, meth, pat, *args):
        if len(set("[]*?") & set(pat)):
            names = fnmatch.filter(self.get_parameter_names(full=True), pat)
        else:
            names = [pat]
        elements = []
        for name in names:
            n = name.split(":")
            if n[0] == "k1":
                elements.append(getattr(self.k1, meth)(":".join(n[1:]), *args))
            elif n[0] == "k2":
                elements.append(getattr(self.k2, meth)(":".join(n[1:]), *args))
        if len(elements) == 0:
            raise ValueError("invalid parameter '{0}'".format(pat))
        return elements

    def freeze_parameter(self, parameter_name):
        self._wildcard_apply("freeze_parameter", parameter_name)

    def thaw_parameter(self, parameter_name):
        self._wildcard_apply("thaw_parameter", parameter_name)

    def get_parameter(self, parameter_name):
        r = self._wildcard_apply("get_parameter", parameter_name)
        if len(r) == 1:
            r = r[0]
        try:
            return float(r)
        except TypeError:
            return np.array(r, dtype=float)

    def set_parameter(self, parameter_name, value):
        self._wildcard_apply("set_parameter", parameter_name, value)

    def get_bounds(self):
        return self.k1.get_bounds() + self.k2.get_bounds()


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
    r"""
    {{ spec.doc | indent(4) }}

    """

    kernel_type = {{ spec.index }}
    stationary = {{ spec.stationary }}
    _parameter_names = [{% for p in spec.params -%}"{{ p }}", {% endfor %}]

    def __init__(self,
                 {% for p in spec.params %}{{ p }}=None,
                 {{ p }}_bounds=None,
                 {%- if p.startswith("ln_") %}
                 {{ p[3:] }}=None,
                 {{ p[3:] }}_bounds=None,{% endif %}
                 {% endfor -%}
                 {% for con in spec.constants %}{{ con.name }}=None,
                 {% endfor -%}
                 {% if spec.stationary -%}
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 {% endif -%}
                 ndim=1,
                 axes=None):
        self._parameter_vector = np.empty({{ spec.params|length }})
        self._parameter_bounds = [(None, None) for _ in xrange({{ spec.params|length }})]
        self._unfrozen = np.ones({{ spec.params|length }}, dtype=bool)

        {% for p in spec.params -%}
        if {{ p }} is None{% if p.startswith("ln_") %} and {{ p[3:] }} is None{% endif %}:
            raise ValueError("missing required parameter '{{ p }}'{% if p.startswith("ln_") %} or '{{ p[3:] }}'{% endif %}")
        {%- if p.startswith("ln_") %}
        elif {{ p }} is None:
            if {{ p[3:] }} <= 0.0:
                raise ValueError("invalid parameter '{{ p[3:] }} <= 0.0'")
            {{ p }} = np.log({{ p[3:] }})
            if {{ p[3:] }}_bounds is not None:
                if len({{ p[3:] }}_bounds) != 2:
                    raise ValueError("invalid bounds for '{{ p[3:] }}'")
                a, b = {{ p[3:] }}_bounds
                {{ p }}_bounds = (
                    None if a is None else np.log(a),
                    None if b is None else np.log(b),
                )
        {%- endif %}
        self.{{ p }} = {{ p }}
        if {{ p }}_bounds is not None:
            if len({{ p }}_bounds) != 2:
                raise ValueError("invalid bounds for '{{ p }}'")
            self._parameter_bounds[{{ loop.index - 1}}] = {{ p }}_bounds
        {% endfor %}
        {% for con in spec.constants %}
        if {{ con.name }} is None:
            raise ValueError("missing required parameter '{{ con.name }}'")
        self.{{ con.name }} = {{ con.name }}
        {% endfor %}
        {% if spec.stationary -%}
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        self.metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                             axes=axes, lower=lower)
        self.ndim = self.metric.ndim
        self.axes = self.metric.axes
        self.block = block
        {%- else -%}
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes
        {%- endif %}

        # Common setup.
        self.dirty = True
        self._kernel = None

    {% if spec.stationary -%}
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
