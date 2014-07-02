# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Sum", "Product", "Kernel",
    "ConstantKernel", "DotProductKernel",
    "CovKernel", "ExpKernel", "ExpSquaredKernel", "RBFKernel",
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

        k(x_i,\,x_j) = c^2

    where :math:`c` is the single parameter ``value``.

    """
    kernel_type = 0

    def __init__(self, value, ndim=1):
        super(ConstantKernel, self).__init__(value, ndim=ndim)

    def __call__(self, x1, x2):
        return self.pars[0] ** 2 + np.sum(np.zeros_like(x1 - x2), axis=-1)


class DotProductKernel(Kernel):
    r"""
    The **dot-product** kernel

    .. math::

        k(x_i,\,x_j) = x_i^{\mathrm{T}} \cdot x_j

    """
    kernel_type = 1

    def __init__(self, ndim=1):
        super(DotProductKernel, self).__init__(ndim=ndim)

    def __call__(self, x1, x2):
        return np.sum(x1 * x2, axis=-1)


class CovKernel(Kernel):

    def __init__(self, cov, ndim=1):
        inds = np.tri(ndim, dtype=bool)
        try:
            l = len(cov)
        except TypeError:
            pars = np.diag(float(cov) * np.ones(ndim))[inds]
        else:
            if l == ndim:
                pars = np.diag(cov)[inds]
            else:
                pars = np.array(cov)
                if l != (ndim*ndim + ndim) / 2:
                    raise ValueError("Dimension mismatch")
        super(CovKernel, self).__init__(*pars, ndim=ndim)

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


class ExpKernel(CovKernel):
    r"""
    The **exponential** kernel.

    .. math::

        k(x_i,\,x_j) = \exp \left ( -\sqrt{\mathbf{r}^\mathrm{T}\,
                                   C^{-1}\,\mathbf{r}} \right )

    """
    kernel_type = 2

    def get_value(self, dx):
        return np.exp(-np.sqrt(dx))


class ExpSquaredKernel(CovKernel):
    r"""
    The **exponential-squared** kernel.

    .. math::

        k(x_i,\,x_j) = \exp \left ( -\frac{1}{2}\,\mathbf{r}^\mathrm{T}\,
                                   C^{-1}\,\mathbf{r} \right )

    """
    kernel_type = 3

    def get_value(self, dx):
        return np.exp(-0.5 * dx)


class RBFKernel(ExpSquaredKernel):
    r"""
    An alias for :class:`ExpSquaredKernel`.

    """


class Matern32Kernel(CovKernel):
    r"""
    The **Matern-3/2** kernel.

    .. math::

        k(r) = \left( 1+\sqrt{3\,r} \right)\,
               \exp \left (-\sqrt{3\,r} \right )

    """
    kernel_type = 6

    def get_value(self, dx):
        r = np.sqrt(3.0 * dx)
        return (1.0 + r) * np.exp(-r)


class Matern52Kernel(CovKernel):
    r"""
    The **Matern-5/2** kernel.

    .. math::

        k(r) = \left( 1+\sqrt{5\,r} + \frac{5\,r}{3} \right)\,
               \exp \left (-\sqrt{5\,r} \right )

    """
    kernel_type = 7

    def get_value(self, dx):
        r = np.sqrt(5.0 * dx)
        return (1.0 + r + r*r / 3.0) * np.exp(-r)


class CosineKernel(Kernel):
    kernel_type = 4

    def __init__(self, period, ndim=1):
        super(CosineKernel, self).__init__(period, ndim=ndim)
        self._omega = 2 * np.pi / np.abs(period)

    def __call__(self, x1, x2):
        return np.cos(self._omega * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)))


class ExpSine2Kernel(Kernel):
    kernel_type = 5

    def __init__(self, gamma, period, ndim=1):
        super(ExpSine2Kernel, self).__init__(gamma, period, ndim=ndim)
        self._gamma = np.abs(gamma)
        self._omega = np.pi / np.abs(period)

    def __call__(self, x1, x2):
        s = np.sin(self._omega * np.sqrt(np.sum((x1 - x2) ** 2, axis=-1)))
        return np.exp(-self._gamma * s**2)
