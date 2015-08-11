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


#
# BASIC KERNELS
#

# def test_custom():
#     def f(x1, x2, p):
#         return np.exp(-0.5 * np.dot(x1, x2) / p[0])

#     def g(x1, x2, p):
#         arg = 0.5 * np.dot(x1, x2) / p[0]
#         return np.exp(-arg) * arg / p[0]

#     def wrong_g(x1, x2, p):
#         arg = 0.5 * np.dot(x1, x2) / p[0]
#         return 10 * np.exp(-arg) * arg / p[0]

#     do_kernel_t(kernels.PythonKernel(f, g, pars=[0.5]))
#     do_kernel_t(kernels.PythonKernel(f, g, pars=[0.1]))

#     try:
#         do_kernel_t(kernels.PythonKernel(f, wrong_g, pars=[0.5]))
#     except AssertionError:
#         pass
#     else:
#         assert False, "This test should fail"


# def test_custom_numerical():
#     def f(x1, x2, p):
#         return np.exp(-0.5 * np.dot(x1, x2) / p[0])
#     do_kernel_t(kernels.PythonKernel(f, pars=[0.5]))
#     do_kernel_t(kernels.PythonKernel(f, pars=[10.0]))


def test_constant():
    do_kernel_t(kernels.ConstantKernel(0.1))
    do_kernel_t(kernels.ConstantKernel(10.0, 2))
    do_kernel_t(kernels.ConstantKernel(5.0, 5))


def test_dot_prod():
    do_kernel_t(kernels.DotProductKernel())
    do_kernel_t(kernels.DotProductKernel(2))
    do_kernel_t(kernels.DotProductKernel(5))


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
    do_stationary_t(kernels.RationalQuadraticKernel, ln_alpha=np.log(1.0))
    do_stationary_t(kernels.RationalQuadraticKernel, ln_alpha=np.log(0.1))
    do_stationary_t(kernels.RationalQuadraticKernel, ln_alpha=np.log(10.0))


#
# PERIODIC KERNELS
#

def test_cosine():
    do_kernel_t(kernels.CosineKernel(1.0))
    do_kernel_t(kernels.CosineKernel(0.5, ndim=2))
    do_kernel_t(kernels.CosineKernel(0.5, ndim=2, axes=1))
    do_kernel_t(kernels.CosineKernel(0.75, ndim=5, axes=[2, 3]))


def test_exp_sine2():
    do_kernel_t(kernels.ExpSine2Kernel(0.4, 1.0))
    do_kernel_t(kernels.ExpSine2Kernel(12., 0.5, ndim=2))
    do_kernel_t(kernels.ExpSine2Kernel(17., 0.5, ndim=2, axes=1))
    do_kernel_t(kernels.ExpSine2Kernel(13.7, -0.75, ndim=5, axes=[2, 3]))
    do_kernel_t(kernels.ExpSine2Kernel(-0.7, 0.75, ndim=5, axes=[2, 3]))
    do_kernel_t(kernels.ExpSine2Kernel(-10, 0.75))


#
# COMBINING KERNELS
#

def test_combine():
    do_kernel_t(12 * kernels.ExpSine2Kernel(0.4, 1.0, ndim=5) + 0.1)
    do_kernel_t(12 * kernels.ExpSquaredKernel(0.4, ndim=3) + 0.1)
