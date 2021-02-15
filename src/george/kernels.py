# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "Kernel", "Sum", "Product",
    "LinearKernel",
    "RationalQuadraticKernel",
    "ExpKernel",
    "LocalGaussianKernel",
    "EmptyKernel",
    "CosineKernel",
    "Matern52Kernel",
    "ExpSine2Kernel",
    "ConstantKernel",
    "ExpSquaredKernel",
    "Matern32Kernel",
    "PolynomialKernel",
    "DotProductKernel",
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


class BaseLinearKernel (Model):
    parameter_names = ("log_gamma2", )


class LinearKernel (Kernel):
    r"""
    The linear regression kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) =
            \frac{(\mathbf{x}_i \cdot \mathbf{x}_j)^P}{\gamma^2}

    :param order:
        The power :math:`P`. This parameter is a *constant*; it is not
        included in the parameter vector.

    :param log_gamma2:
        The scale factor :math:`\gamma^2`.


    """

    kernel_type = 0
    stationary = False

    def __init__(self,
                 log_gamma2=None,
                 order=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if order is None:
            raise ValueError("missing required parameter 'order'")
        self.order = order
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(log_gamma2=log_gamma2, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseLinearKernel(**kwargs)
        super(LinearKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseRationalQuadraticKernel (Model):
    parameter_names = ("log_alpha", )


class RationalQuadraticKernel (Kernel):
    r"""
    This is equivalent to a "scale mixture" of :class:`ExpSquaredKernel`
    kernels with different scale lengths drawn from a gamma distribution.
    See R&W for more info but here's the equation:

    .. math::
        k(r^2) = \left[1 - \frac{r^2}{2\,\alpha} \right]^\alpha

    :param log_alpha:
        The Gamma distribution parameter.


    """

    kernel_type = 1
    stationary = True

    def __init__(self,
                 log_alpha=None,
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block

        kwargs = dict(log_alpha=log_alpha, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseRationalQuadraticKernel(**kwargs)
        super(RationalQuadraticKernel, self).__init__([
            (None, base), ("metric", metric)
        ])

        # Common setup.
        self.dirty = True
    
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
    

class BaseExpKernel (Model):
    parameter_names = ()


class ExpKernel (Kernel):
    r"""
    The exponential kernel is a stationary kernel where the value
    at a given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \exp \left ( -\sqrt{r^2} \right )


    """

    kernel_type = 2
    stationary = True

    def __init__(self,
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseExpKernel(**kwargs)
        super(ExpKernel, self).__init__([
            (None, base), ("metric", metric)
        ])

        # Common setup.
        self.dirty = True
    
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
    

class BaseLocalGaussianKernel (Model):
    parameter_names = ("location", "log_width", )


class LocalGaussianKernel (Kernel):
    r"""
    A local Gaussian kernel.

    .. math::
        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp\left(
            -\frac{(x_i - x_0)^2 + (x_j - x_0)^2}{2\,w} \right))

    :param location:
        The location :math:`x_0` of the Gaussian.

    :param log_width:
        The (squared) width :math:`w` of the Gaussian.


    """

    kernel_type = 3
    stationary = False

    def __init__(self,
                 location=None,
                 log_width=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(location=location, log_width=log_width, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseLocalGaussianKernel(**kwargs)
        super(LocalGaussianKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseEmptyKernel (Model):
    parameter_names = ()


class EmptyKernel (Kernel):
    r"""
    This kernel is a no-op

    """

    kernel_type = 4
    stationary = False

    def __init__(self,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseEmptyKernel(**kwargs)
        super(EmptyKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseCosineKernel (Model):
    parameter_names = ("log_period", )


class CosineKernel (Kernel):
    r"""
    The simplest periodic kernel. This

    .. math::
        k(\mathbf{x}_i,\,\mathbf{x}_j) = \cos\left(
            \frac{2\,\pi\,|x_i - x_j|}{P} \right)

    where the parameter :math:`P` is the period of the oscillation. This
    kernel should probably always be multiplied be a stationary kernel
    (e.g. :class:`ExpSquaredKernel`) to allow quasi-periodic variations.

    :param log_period:
        The period of the oscillation.


    """

    kernel_type = 5
    stationary = False

    def __init__(self,
                 log_period=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(log_period=log_period, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseCosineKernel(**kwargs)
        super(CosineKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseMatern52Kernel (Model):
    parameter_names = ()


class Matern52Kernel (Kernel):
    r"""
    The Matern-5/2 kernel is stationary kernel where the value at a
    given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \left( 1+\sqrt{5\,r^2}+ \frac{5\,r^2}{3} \right)\,
                 \exp \left (-\sqrt{5\,r^2} \right )


    """

    kernel_type = 6
    stationary = True

    def __init__(self,
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseMatern52Kernel(**kwargs)
        super(Matern52Kernel, self).__init__([
            (None, base), ("metric", metric)
        ])

        # Common setup.
        self.dirty = True
    
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
    

class BaseExpSine2Kernel (Model):
    parameter_names = ("gamma", "log_period", )


class ExpSine2Kernel (Kernel):
    r"""
    The exp-sine-squared kernel is a commonly used periodic kernel. Unlike
    the :class:`CosineKernel`, this kernel never has negative covariance
    which might be useful for your problem. Here's the equation:

    .. math::
        k(\mathbf{x}_i,\,\mathbf{x}_j) =
            \exp \left( -\Gamma\,\sin^2\left[
                \frac{\pi}{P}\,\left|x_i-x_j\right|
            \right] \right)

    :param gamma:
        The scale :math:`\Gamma` of the correlations.

    :param log_period:
        The log of the period :math:`P` of the oscillation (in the same units
        as :math:`\mathbf{x}`).


    """

    kernel_type = 7
    stationary = False

    def __init__(self,
                 gamma=None,
                 log_period=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(gamma=gamma, log_period=log_period, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseExpSine2Kernel(**kwargs)
        super(ExpSine2Kernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseConstantKernel (Model):
    parameter_names = ("log_constant", )


class ConstantKernel (Kernel):
    r"""
    This kernel returns the constant

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = c

    where :math:`c` is a parameter.

    :param log_constant:
        The log of :math:`c` in the above equation.


    """

    kernel_type = 8
    stationary = False

    def __init__(self,
                 log_constant=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(log_constant=log_constant, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseConstantKernel(**kwargs)
        super(ConstantKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseExpSquaredKernel (Model):
    parameter_names = ()


class ExpSquaredKernel (Kernel):
    r"""
    The exponential-squared kernel is a stationary kernel where the value
    at a given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \exp \left ( -\frac{r^2}{2} \right )


    """

    kernel_type = 9
    stationary = True

    def __init__(self,
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseExpSquaredKernel(**kwargs)
        super(ExpSquaredKernel, self).__init__([
            (None, base), ("metric", metric)
        ])

        # Common setup.
        self.dirty = True
    
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
    

class BaseMatern32Kernel (Model):
    parameter_names = ()


class Matern32Kernel (Kernel):
    r"""
    The Matern-3/2 kernel is stationary kernel where the value at a
    given radius :math:`r^2` is given by:

    .. math::

        k(r^2) = \left( 1+\sqrt{3\,r^2} \right)\,
                 \exp \left (-\sqrt{3\,r^2} \right )


    """

    kernel_type = 10
    stationary = True

    def __init__(self,
                 metric=None,
                 metric_bounds=None,
                 lower=True,
                 block=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if metric is None:
            raise ValueError("missing required parameter 'metric'")
        metric = Metric(metric, bounds=metric_bounds, ndim=ndim,
                        axes=axes, lower=lower)
        self.ndim = metric.ndim
        self.axes = metric.axes
        self.block = block

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseMatern32Kernel(**kwargs)
        super(Matern32Kernel, self).__init__([
            (None, base), ("metric", metric)
        ])

        # Common setup.
        self.dirty = True
    
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
    

class BasePolynomialKernel (Model):
    parameter_names = ("log_sigma2", )


class PolynomialKernel (Kernel):
    r"""
    A polynomial kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) =
            (\mathbf{x}_i \cdot \mathbf{x}_j + \sigma^2)^P

    :param order:
        The power :math:`P`. This parameter is a *constant*; it is not
        included in the parameter vector.

    :param log_sigma2:
        The variance :math:`\sigma^2 > 0`.


    """

    kernel_type = 11
    stationary = False

    def __init__(self,
                 log_sigma2=None,
                 order=None,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        if order is None:
            raise ValueError("missing required parameter 'order'")
        self.order = order
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict(log_sigma2=log_sigma2, )
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BasePolynomialKernel(**kwargs)
        super(PolynomialKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    

class BaseDotProductKernel (Model):
    parameter_names = ()


class DotProductKernel (Kernel):
    r"""
    The dot product kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j

    with no parameters.


    """

    kernel_type = 12
    stationary = False

    def __init__(self,
                 bounds=None,
                 ndim=1,
                 axes=None):
        
        self.subspace = Subspace(ndim, axes=axes)
        self.ndim = self.subspace.ndim
        self.axes = self.subspace.axes

        kwargs = dict()
        if bounds is not None:
            kwargs["bounds"] = bounds
        base = BaseDotProductKernel(**kwargs)
        super(DotProductKernel, self).__init__([
            (None, base), 
        ])

        # Common setup.
        self.dirty = True
    
