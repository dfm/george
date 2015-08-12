# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = [
]

import numpy as np

from .. import kernels
from ..gp import GP


def test_dtype(seed=123):
    np.random.seed(seed)
    kernel = 0.1 * kernels.ExpSquaredKernel(1.5)
    kernel.pars = [1, 2]
    gp = GP(kernel)
    x = np.random.rand(100)
    gp.compute(x, 1e-2)


def do_kernel_t(kernel, N=20, seed=123, eps=1.32e-6):
    """
    Test that both the Python and C++ kernels return the same matrices.

    """
    np.random.seed(seed)
    t1 = np.random.randn(N, kernel.ndim)

    v = kernel.get_vector()

    # Check the symmetric gradients.
    g1 = kernel.get_gradient(t1)
    for i in range(len(kernel)):
        # Compute the centered finite difference approximation to the gradient.
        v[i] += eps
        kernel.set_vector(v)
        kp = kernel.get_value(t1)
        v[i] -= 2*eps
        kernel.set_vector(v)
        km = kernel.get_value(t1)
        v[i] += eps
        kernel.set_vector(v)
        g0 = 0.5 * (kp - km) / eps
        assert np.allclose(g1[:, :, i], g0), \
            "Gradient computation failed in dimension {0}".format(i)


def test_constant():
    do_kernel_t(kernels.ConstantKernel(constant=0.1))
    do_kernel_t(kernels.ConstantKernel(constant=10.0, ndim=2))
    do_kernel_t(kernels.ConstantKernel(constant=5.0, ndim=5))


def test_dot_prod():
    do_kernel_t(kernels.DotProductKernel())
    do_kernel_t(kernels.DotProductKernel(ndim=2))
    do_kernel_t(kernels.DotProductKernel(ndim=5, axes=0))


#
# STATIONARY KERNELS
#

def do_stationary_t(kernel_type, **kwargs):
    def build_kernel(metric, **more):
        kws = dict(kwargs, **more)
        return kernel_type(metric=metric, **kws)

    kernel = build_kernel(0.1)
    do_kernel_t(kernel)

    kernel = build_kernel(1.0)
    do_kernel_t(kernel)

    kernel = build_kernel(10.0)
    do_kernel_t(kernel)

    kernel = build_kernel([1.0, 0.1, 10.0], ndim=3)
    do_kernel_t(kernel)

    kernel = build_kernel(1.0, ndim=3)
    do_kernel_t(kernel)

    try:
        kernel = build_kernel([1.0, 0.1, 10.0, 500], ndim=3)
    except:
        pass
    else:
        assert False, "This test should fail"

    kernel = build_kernel(1.0, ndim=3, axes=2)
    do_kernel_t(kernel)


def test_exp():
    do_stationary_t(kernels.ExpKernel)


def test_exp_squared():
    do_stationary_t(kernels.ExpSquaredKernel)


def test_matern32():
    do_stationary_t(kernels.Matern32Kernel)


def test_matern52():
    do_stationary_t(kernels.Matern52Kernel)


def test_rational_quadratic():
    do_stationary_t(kernels.RationalQuadraticKernel, alpha=1.0)
    do_stationary_t(kernels.RationalQuadraticKernel, alpha=0.1)
    do_stationary_t(kernels.RationalQuadraticKernel, alpha=10.0)


def test_cosine():
    do_kernel_t(kernels.CosineKernel(period=1.0))
    do_kernel_t(kernels.CosineKernel(period=0.5, ndim=2))
    do_kernel_t(kernels.CosineKernel(period=0.5, ndim=2, axes=1))
    do_kernel_t(kernels.CosineKernel(period=0.75, ndim=5, axes=[2, 3]))


def test_exp_sine2():
    do_kernel_t(kernels.ExpSine2Kernel(gamma=0.4, period=1.0))
    do_kernel_t(kernels.ExpSine2Kernel(gamma=12., period=0.5, ndim=2))
    do_kernel_t(kernels.ExpSine2Kernel(gamma=17., period=0.5, ndim=2, axes=1))
    do_kernel_t(kernels.ExpSine2Kernel(gamma=13.7, ln_period=-0.75, ndim=5,
                                       axes=[2, 3]))
    do_kernel_t(kernels.ExpSine2Kernel(gamma=-0.7, period=0.75, ndim=5,
                                       axes=[2, 3]))
    do_kernel_t(kernels.ExpSine2Kernel(gamma=-10, period=0.75))


def test_combine():
    do_kernel_t(12 * kernels.ExpSine2Kernel(gamma=0.4, period=1.0, ndim=5))
    do_kernel_t(12 * kernels.ExpSquaredKernel(0.4, ndim=3) + 0.1)
