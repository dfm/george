# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from george import kernels, GP


def test_dtype(seed=123):
    np.random.seed(seed)
    kernel = 0.1 * kernels.ExpSquaredKernel(1.5)
    kernel.pars = [1, 2]
    gp = GP(kernel)
    x = np.random.rand(100)
    gp.compute(x, 1e-2)

kernels_to_test = [
    kernels.ConstantKernel(log_constant=0.1),
    kernels.ConstantKernel(log_constant=10.0, ndim=2),
    kernels.ConstantKernel(log_constant=5.0, ndim=5),

    kernels.DotProductKernel(),
    kernels.DotProductKernel(ndim=2),
    kernels.DotProductKernel(ndim=5, axes=0),

    kernels.CosineKernel(log_period=1.0),
    kernels.CosineKernel(log_period=0.5, ndim=2),
    kernels.CosineKernel(log_period=0.5, ndim=2, axes=1),
    kernels.CosineKernel(log_period=0.75, ndim=5, axes=[2, 3]),

    kernels.ExpSine2Kernel(gamma=0.4, log_period=1.0),
    kernels.ExpSine2Kernel(gamma=12., log_period=0.5, ndim=2),
    kernels.ExpSine2Kernel(gamma=17., log_period=0.5, ndim=2, axes=1),
    kernels.ExpSine2Kernel(gamma=13.7, log_period=-0.75, ndim=5, axes=[2, 3]),
    kernels.ExpSine2Kernel(gamma=-0.7, log_period=0.75, ndim=5, axes=[2, 3]),
    kernels.ExpSine2Kernel(gamma=-10, log_period=0.75),

    kernels.LocalGaussianKernel(log_width=0.5, location=1.0),
    kernels.LocalGaussianKernel(log_width=0.1, location=0.5, ndim=2),
    kernels.LocalGaussianKernel(log_width=1.5, location=-0.5, ndim=2, axes=1),
    kernels.LocalGaussianKernel(log_width=2.0, location=0.75, ndim=5,
                                axes=[2, 3]),

    kernels.LinearKernel(order=0, log_gamma2=0.0),
    kernels.LinearKernel(order=2, log_gamma2=0.0),
    kernels.LinearKernel(order=2, log_gamma2=0.0),
    kernels.LinearKernel(order=5, log_gamma2=1.0, ndim=2),
    kernels.LinearKernel(order=3, log_gamma2=-1.0, ndim=5, axes=2),
    kernels.LinearKernel(order=0, log_gamma2=0.0) +
    kernels.LinearKernel(order=1, log_gamma2=-1.0) +
    kernels.LinearKernel(order=2, log_gamma2=-2.0),

    kernels.PolynomialKernel(order=0, log_sigma2=-10.0),
    kernels.PolynomialKernel(order=2, log_sigma2=-10.0),
    kernels.PolynomialKernel(order=2, log_sigma2=0.0),
    kernels.PolynomialKernel(order=5, log_sigma2=1.0, ndim=2),
    kernels.PolynomialKernel(order=3, log_sigma2=-1.0, ndim=5, axes=2),

    12. * kernels.ExpSine2Kernel(gamma=0.4, log_period=1.0, ndim=5),
    12. * kernels.ExpSquaredKernel(0.4, ndim=3) + 0.1,
]

@pytest.mark.parametrize("kernel", kernels_to_test)
def test_kernel(kernel, N=20, seed=123, eps=1.32e-6):
    np.random.seed(seed)
    t1 = np.random.randn(N, kernel.ndim)
    kernel.test_gradient(t1, eps=eps)
    kernel.test_gradient(t1, t1[:1], eps=eps)


@pytest.mark.parametrize("kernel", kernels_to_test)
def test_x_gradient_kernel(kernel, N=20, seed=123, eps=1.32e-6):
    np.random.seed(seed)
    t1 = np.random.randn(N, kernel.ndim)
    kernel.test_x1_gradient(t1, eps=eps)
    kernel.test_x1_gradient(t1, np.array(t1[:1]), eps=eps)
    kernel.test_x2_gradient(t1, eps=eps)
    kernel.test_x2_gradient(np.array(t1[:1]), t1, eps=eps)


stationary_kernels = [
    (kernels.ExpKernel, {}),
    (kernels.ExpSquaredKernel, {}),
    (kernels.Matern32Kernel, {}),
    (kernels.Matern52Kernel, {}),
    (kernels.RationalQuadraticKernel, dict(log_alpha=np.log(1.0))),
    (kernels.RationalQuadraticKernel, dict(log_alpha=np.log(0.1))),
    (kernels.RationalQuadraticKernel, dict(log_alpha=np.log(10.0))),
]

@pytest.mark.parametrize("kernel_type,kwargs", stationary_kernels)
def test_stationary(kernel_type, kwargs):
    def build_kernel(metric, **more):
        kws = dict(kwargs, **more)
        return kernel_type(metric=metric, **kws)

    kernel = build_kernel(0.1)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    kernel = build_kernel(1.0)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    kernel = build_kernel(10.0)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    kernel = build_kernel([1.0, 0.1, 10.0], ndim=3)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    kernel = build_kernel(1.0, ndim=3)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    with pytest.raises(ValueError):
        kernel = build_kernel([1.0, 0.1, 10.0, 500], ndim=3)

    kernel = build_kernel(1.0, ndim=3, axes=2)
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)

    kernel = build_kernel(1.0, ndim=3, axes=2, block=(-0.1, 0.1))
    test_kernel(kernel)
    test_x_gradient_kernel(kernel)
