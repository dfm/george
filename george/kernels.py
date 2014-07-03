# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Sum", "Product", "Kernel",
    "ConstantKernel", "DotProductKernel",
    "RadialKernel", "ExpKernel", "ExpSquaredKernel", "RBFKernel",
    "CosineKernel", "ExpSine2Kernel",
    "Matern32Kernel", "Matern52Kernel",
]

import numpy as np


class Kernel(object):
    """
    The abstract kernel type.

    """

    is_kernel = True
    kernel_type = -1

    def __init__(self, *pars, **kwargs):
        self.ndim = kwargs.get("ndim", 1)
        self.pars = np.array(pars)

    def __len__(self):
        return len(self.pars)

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(ConstantKernel(np.sqrt(np.abs(float(b))),
                                      ndim=self.ndim), self)
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(ConstantKernel(np.sqrt(np.abs(float(b))),
                                          ndim=self.ndim), self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(Kernel):
    is_kernel = False
    operator_type = -1

    def __init__(self, k1, k2):
        if k1.ndim != k2.ndim:
            raise ValueError("Dimension mismatch")
        self.k1 = k1
        self.k2 = k2
        self.ndim = k1.ndim


class Sum(_operator):
    is_kernel = False
    operator_type = 0

    def __call__(self, x1, x2):
        return self.k1(x1, x2) + self.k2(x1, x2)


class Product(_operator):
    is_kernel = False
    operator_type = 1

    def __call__(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)


class ConstantKernel(Kernel):
    r"""
    This kernel returns the constant

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = c^2

    where :math:`c` is a parameter.

    :param value:
        The constant value :math:`c` in the above equation.

    """
    kernel_type = 0

    def __init__(self, value, ndim=1):
        super(ConstantKernel, self).__init__(value, ndim=ndim)

    def __call__(self, x1, x2):
        return self.pars[0] ** 2 + np.sum(np.zeros_like(x1 - x2), axis=-1)


class DotProductKernel(Kernel):
    r"""
    The dot-product kernel takes the form

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \mathbf{x}_i^{\mathrm{T}} \cdot
                                         \mathbf{x}_j

    """
    kernel_type = 1

    def __init__(self, ndim=1):
        super(DotProductKernel, self).__init__(ndim=ndim)

    def __call__(self, x1, x2):
        return np.sum(x1 * x2, axis=-1)


class RadialKernel(Kernel):
    """
    This abstract base class implements a radial kernel in an arbitrary
    metric.  The metric is specified as a matrix :math:`C` where the
    radius :math:`{r_{ij}}^2` is

    .. math::

        {r_{ij}}^2 = (\mathbf{x}_i - \mathbf{x}_j)^\mathrm{T}\,
                     C^{-1}\,(\mathbf{x}_i - \mathbf{x}_j)

    :param metric:
        There are a few different ways that you can specify the metric:

        1. if ``metric`` is a scalar, the metric is assumed isotropic with an
           axis-aligned variance of ``metric`` in each dimension,
        2. if ``metric`` is one-dimensional, it is assumed to specify the
           axis-aligned variances in each dimension, and
        3. if ``metric`` is two-dimensional, it is assumed to give the full
           matrix :math:`C`.

    **Note:**
    Subclasses should implement the :func:`get_value` method to give
    the value of the kernel at a given radius and this class will deal with
    the metric.

    """

    def __init__(self, metric, ndim=1):
        inds = np.tri(ndim, dtype=bool)
        try:
            l = len(metric)
        except TypeError:
            pars = np.diag(float(metric) * np.ones(ndim))[inds]
        else:
            if l == ndim:
                pars = np.diag(metric)[inds]
            else:
                pars = np.array(metric)
                if l != (ndim*ndim + ndim) / 2:
                    raise ValueError("Dimension mismatch")
        super(RadialKernel, self).__init__(*pars, ndim=ndim)

        # Build the covariance matrix.
        self.matrix = np.zeros((self.ndim, self.ndim))
        self.matrix[np.tril_indices_from(self.matrix)] = self.pars
        self.matrix += self.matrix.T
        self.matrix[np.diag_indices_from(self.matrix)] *= 0.5

    def __call__(self, x1, x2):
        dx = x1 - x2
        dxf = dx.reshape((-1, self.ndim)).T
        r = np.sum(dxf * np.linalg.solve(self.matrix, dxf), axis=0)
        r = r.reshape(dx.shape[:-1])
        return self.get_value(r)

    def get_value(self, r):
        raise NotImplementedError("Subclasses must implement this method.")


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
    kernel_type = 2

    def get_value(self, dx):
        return np.exp(-np.sqrt(dx))


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
    kernel_type = 3

    def get_value(self, dx):
        return np.exp(-0.5 * dx)


class RBFKernel(ExpSquaredKernel):
    r"""
    An alias for :class:`ExpSquaredKernel`.

    """


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
    kernel_type = 6

    def get_value(self, dx):
        r = np.sqrt(3.0 * dx)
        return (1.0 + r) * np.exp(-r)


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
    kernel_type = 7

    def get_value(self, dx):
        r = np.sqrt(5.0 * dx)
        return (1.0 + r + r*r / 3.0) * np.exp(-r)


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
    kernel_type = 4

    def __init__(self, period, ndim=1):
        super(CosineKernel, self).__init__(period, ndim=ndim)
        self._omega = 2 * np.pi / np.abs(period)

    def __call__(self, x1, x2):
        return np.cos(self._omega * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)))


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

    **Note:**
    A shortcoming of this kernel is that it currently only accepts a single
    period and scale so it's not very applicable to problems with input
    dimension larger than one.

    """
    kernel_type = 5

    def __init__(self, gamma, period, ndim=1):
        super(ExpSine2Kernel, self).__init__(gamma, period, ndim=ndim)
        self._gamma = np.abs(gamma)
        self._omega = np.pi / np.abs(period)

    def __call__(self, x1, x2):
        s = np.sin(self._omega * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)))
        return np.exp(-self._gamma * s**2)
