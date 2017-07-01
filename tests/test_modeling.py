# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "test_constant_mean",
    "test_callable_mean",
    "test_gp_mean",
    "test_gp_white_noise",
    "test_gp_callable_mean",
    "test_parameters",
    "test_bounds",
]

import pytest
import numpy as np

from george import GP, kernels
from george.utils import check_gradient
from george.modeling import Model, ConstantModel, CallableModel


def test_constant_mean():
    m = ConstantModel(5.0)
    check_gradient(m, np.zeros(4))


def test_callable_mean():
    m = CallableModel(lambda x: 5.0 * x)
    check_gradient(m, np.zeros(4))


def test_gp_mean(N=50, seed=1234):
    np.random.seed(seed)
    x = np.random.uniform(0, 5)
    y = 5 + np.sin(x)
    gp = GP(10. * kernels.ExpSquaredKernel(1.3),
            mean=5.0, fit_mean=True)
    gp.compute(x)
    check_gradient(gp, y)


def test_gp_callable_mean(N=50, seed=1234):
    np.random.seed(seed)
    x = np.random.uniform(0, 5)
    y = 5 + np.sin(x)
    mean = CallableModel(lambda x: 5.0*x)
    gp = GP(10. * kernels.ExpSquaredKernel(1.3), mean=mean)
    gp.compute(x)
    check_gradient(gp, y)


def test_gp_white_noise(N=50, seed=1234):
    np.random.seed(seed)
    x = np.random.uniform(0, 5)
    y = 5 + np.sin(x)
    gp = GP(10. * kernels.ExpSquaredKernel(1.3),
            mean=5.0, fit_mean=True,
            white_noise=0.1, fit_white_noise=True)
    gp.compute(x)
    check_gradient(gp, y)


class LinearWhiteNoise(Model):
    parameter_names = ("m", "b")

    def get_value(self, x):
        return self.m * x + self.b

    @Model.parameter_sort
    def compute_gradient(self, x):
        return dict(m=x, b=np.ones(len(x)))


def test_gp_callable_white_noise(N=50, seed=1234):
    np.random.seed(seed)
    x = np.random.uniform(0, 5)
    y = 5 + np.sin(x)
    gp = GP(10. * kernels.ExpSquaredKernel(1.3), mean=5.0,
            white_noise=LinearWhiteNoise(-6, 0.01),
            fit_white_noise=True)
    gp.compute(x)
    check_gradient(gp, y)

    gp.freeze_parameter("white_noise:m")
    check_gradient(gp, y)


def test_parameters():
    kernel = 10 * kernels.ExpSquaredKernel(1.0)
    kernel += 0.5 * kernels.RationalQuadraticKernel(log_alpha=0.1, metric=5.0)
    gp = GP(kernel, white_noise=LinearWhiteNoise(1.0, 0.1))

    n = len(gp.get_parameter_vector())
    assert n == len(gp.get_parameter_names())
    assert n - 2 == len(kernel.get_parameter_names())

    gp.freeze_parameter(gp.get_parameter_names()[0])
    assert n - 1 == len(gp.get_parameter_names())
    assert n - 1 == len(gp.get_parameter_vector())

    gp.freeze_all_parameters()
    assert len(gp.get_parameter_names()) == 0
    assert len(gp.get_parameter_vector()) == 0

    gp.kernel.thaw_all_parameters()
    gp.white_noise.thaw_all_parameters()
    assert n == len(gp.get_parameter_vector())
    assert n == len(gp.get_parameter_names())

    assert np.allclose(kernel[0], np.log(10.))


def test_bounds():
    kernel = 10 * kernels.ExpSquaredKernel(1.0, metric_bounds=[(None, 4.0)])
    kernel += 0.5 * kernels.RationalQuadraticKernel(log_alpha=0.1, metric=5.0)
    gp = GP(kernel, white_noise=LinearWhiteNoise(1.0, 0.1))

    # Test bounds length.
    assert len(gp.get_parameter_bounds()) == len(gp.get_parameter_vector())
    gp.freeze_all_parameters()
    gp.thaw_parameter("white_noise:m")
    assert len(gp.get_parameter_bounds()) == len(gp.get_parameter_vector())

    # Test invalid bounds specification.
    with pytest.raises(ValueError):
        kernels.ExpSine2Kernel(gamma=0.1, log_period=5.0, bounds=[10.0])
