# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Sum", "Product", "Kernel",
    "ConstantKernel", "WhiteKernel", "DotProductKernel",
    "RadialKernel", "ExpKernel", "ExpSquaredKernel",
    "CosineKernel", "ExpSine2Kernel",
    "Matern32Kernel", "Matern52Kernel",

    "PythonKernel",
]

import numpy as np
from functools import partial

from ._kernels import CythonKernel
from .utils import numerical_gradient


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
            return Sum(ConstantKernel(float(b), ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(float(b), ndim=self.ndim), self)
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


class ConstantKernel(Kernel):
    r"""
    This kernel returns the constant

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = c

    where :math:`c` is a parameter.

    :param value:
        The constant value :math:`c` in the above equation.

    """
    kernel_type = 0

    def __init__(self, value, ndim=1):
        super(ConstantKernel, self).__init__(value, ndim=ndim)


class WhiteKernel(Kernel):
    r"""
    This kernel returns constant along the diagonal.

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = c \, \delta_{ij}

    where :math:`c` is the parameter.

    :param value:
        The constant value :math:`c` in the above equation.

    """
    kernel_type = 1

    def __init__(self, value, ndim=1):
        super(WhiteKernel, self).__init__(value, ndim=ndim)


class DotProductKernel(Kernel):
    r"""
    The dot-product kernel takes the form

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \mathbf{x}_i^{\mathrm{T}} \cdot
                                         \mathbf{x}_j

    """
    kernel_type = 2

    def __init__(self, ndim=1):
        super(DotProductKernel, self).__init__(ndim=ndim)


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


class ExpKernel(RadialKernel):
    r"""
    The exponential kernel is a :class:`RadialKernel` where the value at a
    given radius :math:`r^2` is given by:

    .. math::

        k({r_{ij}}) = \exp \left ( -|r| \right )

    :param metric:
        The custom metric specified as described in the :class:`RadialKernel`
        description.

    """
    kernel_type = 3


class ExpSquaredKernel(RadialKernel):
    r"""
    The exponential-squared kernel is a :class:`RadialKernel` where the value
    at a given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \exp \left ( -\frac{r^2}{2} \right )

    :param metric:
        The custom metric specified as described in the :class:`RadialKernel`
        description.

    """
    kernel_type = 4


class Matern32Kernel(RadialKernel):
    r"""
    The Matern-3/2 kernel is a :class:`RadialKernel` where the value at a
    given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \left( 1+\sqrt{3\,r^2} \right)\,
                 \exp \left (-\sqrt{3\,r^2} \right )

    :param metric:
        The custom metric specified as described in the :class:`RadialKernel`
        description.

    """
    kernel_type = 5


class Matern52Kernel(RadialKernel):
    r"""
    The Matern-5/2 kernel is a :class:`RadialKernel` where the value at a
    given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \left( 1+\sqrt{5\,r^2} + \frac{5\,r^2}{3} \right)\,
                 \exp \left (-\sqrt{5\,r^2} \right )

    :param metric:
        The custom metric specified as described in the :class:`RadialKernel`
        description.

    """
    kernel_type = 6


class RationalQuadraticKernel(RadialKernel):
    r"""
    TODO: document this kernel.

    """
    kernel_type = 7

    def __init__(self, alpha, metric, ndim=1, **kwargs):
        super(RationalQuadraticKernel, self).__init__(metric, extra=[alpha],
                                                      ndim=ndim, **kwargs)


class CosineKernel(Kernel):
    r"""
    The cosine kernel is given by:

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) =
            \cos\left(\frac{2\,\pi}{P}\,\left|x_i-x_j\right| \right)

    where :math:`P` is the period.

    :param period:
        The period :math:`P` of the oscillation (in the same units as
        :math:`\mathbf{x}`).

    **Note:**
    A shortcoming of this kernel is that it currently only accepts a single
    period so it's not very applicable to problems with input dimension larger
    than one.

    """
    kernel_type = 8

    def __init__(self, period, ndim=1, dim=0):
        super(CosineKernel, self).__init__(period, ndim=ndim)
        assert dim < self.ndim, "Invalid dimension"
        self.dim = dim


class ExpSine2Kernel(Kernel):
    r"""
    The exp-sine-squared kernel is used to model stellar rotation and *might*
    be applicable in some other contexts. It is given by the equation:

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) =
            \sin \left( -\Gamma\,\sin^2\left[
                \frac{\pi}{P}\,\left|x_i-x_j\right|
            \right] \right)

    where :math:`\Gamma` is the "scale" of the correlation and :math:`P` is
    the period of the oscillation measured in the same units as
    :math:`\mathbf{x}`.

    :param gamma:
        The scale :math:`\Gamma` of the correlations.

    :param period:
        The period :math:`P` of the oscillation (in the same units as
        :math:`\mathbf{x}`).

    :param dim: (optional)
        The dimension along which this kernel should apply. By default, this
        will be the zero-th axis.

    """
    kernel_type = 9

    def __init__(self, gamma, period, ndim=1, dim=0):
        super(ExpSine2Kernel, self).__init__(gamma, period, ndim=ndim)
        assert dim < self.ndim, "Invalid dimension"
        self.dim = dim


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
