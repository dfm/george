# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "test_dtype",

    "test_constant", "test_dot_prod", "test_cosine", "test_exp_sine2",
    "test_local", "test_local", "test_polynomial",

    "test_exp", "test_exp_squared", "test_matern32", "test_matern52",
    "test_rational_quadratic", "test_combine",
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
    np.random.seed(seed)
    t1 = np.random.randn(N, kernel.ndim)
    kernel.test_gradient(t1, eps=eps)


def test_constant():
    do_kernel_t(kernels.ConstantKernel(constant=0.1))
    do_kernel_t(kernels.ConstantKernel(constant=10.0, ndim=2))
    do_kernel_t(kernels.ConstantKernel(constant=5.0, ndim=5))


def test_dot_prod():
    do_kernel_t(kernels.DotProductKernel())
    do_kernel_t(kernels.DotProductKernel(ndim=2))
    do_kernel_t(kernels.DotProductKernel(ndim=5, axes=0))


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


def test_local():
    do_kernel_t(kernels.LocalGaussianKernel(width=0.5, location=1.0))
    do_kernel_t(kernels.LocalGaussianKernel(width=0.1, location=0.5, ndim=2))
    do_kernel_t(kernels.LocalGaussianKernel(width=1.5, location=-0.5, ndim=2,
                                            axes=1))
    do_kernel_t(kernels.LocalGaussianKernel(width=2.0, location=0.75, ndim=5,
                                            axes=[2, 3]))


def test_linear():
    do_kernel_t(kernels.LinearKernel(order=0, ln_gamma2=0.0))
    do_kernel_t(kernels.LinearKernel(order=2, ln_gamma2=0.0))
    do_kernel_t(kernels.LinearKernel(order=2, ln_gamma2=0.0))
    do_kernel_t(kernels.LinearKernel(order=5, ln_gamma2=1.0, ndim=2))
    do_kernel_t(kernels.LinearKernel(order=3, ln_gamma2=-1.0, ndim=5,
                                     axes=2))
    k = kernels.LinearKernel(order=0, ln_gamma2=0.0)
    k += kernels.LinearKernel(order=1, ln_gamma2=-1.0)
    k += kernels.LinearKernel(order=2, ln_gamma2=-2.0)
    do_kernel_t(k)


def test_polynomial():
    do_kernel_t(kernels.PolynomialKernel(order=0, ln_sigma2=-10.0))
    do_kernel_t(kernels.PolynomialKernel(order=2, ln_sigma2=-10.0))
    do_kernel_t(kernels.PolynomialKernel(order=2, ln_sigma2=0.0))
    do_kernel_t(kernels.PolynomialKernel(order=5, ln_sigma2=1.0, ndim=2))
    do_kernel_t(kernels.PolynomialKernel(order=3, ln_sigma2=-1.0, ndim=5,
                                         axes=2))


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

    kernel = build_kernel(1.0, ndim=3, axes=2, block=(-0.1, 0.1))
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


def test_combine():
    do_kernel_t(12 * kernels.ExpSine2Kernel(gamma=0.4, period=1.0, ndim=5))
    do_kernel_t(12 * kernels.ExpSquaredKernel(0.4, ndim=3) + 0.1)


def test_grp():
    do_kernel_t(kernels.GRPKernel(qfactor=1.0, amplitude=10.0))
    do_kernel_t(kernels.GRPKernel(qfactor=1.0, amplitude=3.0, ndim=2))
    do_kernel_t(kernels.GRPKernel(qfactor=5.0, amplitude=17.5, ndim=5))

def test_grp_periodic():
    do_kernel_t(kernels.GRPPeriodicKernel(qfactor=17.0, amplitude=5.0,
                                          frequency=3.0))
    do_kernel_t(kernels.GRPPeriodicKernel(qfactor=1.0, amplitude=10.0,
                                          frequency=1.0, ndim=2))
