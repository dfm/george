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

import numpy as np

from ..gp import GP
from .. import kernels
from ..models import ConstantModel, CallableModel
from ..modeling import check_gradient, ModelingMixin


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
    gp = GP(10. * kernels.ExpSquaredKernel(1.3),
            mean=lambda x: 5.0*x, fit_mean=True)
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


class LinearWhiteNoise(ModelingMixin):

    def __init__(self, m, b):
        super(LinearWhiteNoise, self).__init__(m=m, b=b)

    def get_value(self, x):
        return self.m * x + self.b

    @ModelingMixin.parameter_sort
    def get_gradient(self, x):
        return dict(m=x, b=np.ones(len(x)))

    @ModelingMixin.parameter_sort
    def get_bounds(self):
        return dict(m=(-50.0, 10.0), b=(None, 10.0))


def test_gp_callable_white_noise(N=50, seed=1234):
    np.random.seed(seed)
    x = np.random.uniform(0, 5)
    y = 5 + np.sin(x)
    gp = GP(10. * kernels.ExpSquaredKernel(1.3), mean=5.0,
            white_noise=LinearWhiteNoise(-6, 0.01),
            fit_white_noise=True)
    gp.compute(x)
    check_gradient(gp, y)

    gp.freeze_parameter("white:m")
    check_gradient(gp, y)


def test_parameters():
    kernel = 10 * kernels.ExpSquaredKernel(1.0)
    kernel += 0.5 * kernels.RationalQuadraticKernel(alpha=0.1, metric=5.0)
    gp = GP(kernel, white_noise=LinearWhiteNoise(1.0, 0.1),
            fit_white_noise=True)

    n = len(gp.get_vector())
    assert n == len(gp.get_parameter_names())
    assert n - 2 == len(kernel.get_parameter_names())

    gp.freeze_parameter(gp.get_parameter_names()[0])
    assert n - 1 == len(gp.get_parameter_names())
    assert n - 1 == len(gp.get_vector())

    gp.freeze_parameter("*")
    assert len(gp.get_parameter_names()) == 0
    assert len(gp.get_vector()) == 0

    gp.thaw_parameter("*")
    assert n == len(gp.get_vector())
    assert n == len(gp.get_parameter_names())

    assert np.allclose(kernel["*constant"], np.log([10., 0.5]))
    assert np.allclose(kernel[0], np.log(10.))
    assert np.allclose(kernel[[0, 1]], np.log([10., 1.0]))

    gp.freeze_parameter("kernel:*constant")
    assert n-2 == len(gp.get_vector())


def test_bounds():
    kernel = 10 * kernels.ExpSquaredKernel(1.0, metric_bounds=(None, 4.0))
    kernel += 0.5 * kernels.RationalQuadraticKernel(alpha=0.1, metric=5.0)
    gp = GP(kernel, white_noise=LinearWhiteNoise(1.0, 0.1),
            fit_white_noise=True)

    # Test bounds length.
    assert len(gp.get_bounds()) == len(gp.get_vector())
    gp.freeze_parameter("*")
    gp.thaw_parameter("white:m")
    assert len(gp.get_bounds()) == len(gp.get_vector())

    # Test out of bounds.
    (a, b), = gp.get_bounds()
    try:
        gp.set_vector([a - 1.0])
    except ValueError:
        pass
    else:
        assert False, "out of bounds should fail"

    # Test bounds conversions.
    k = kernels.ExpSine2Kernel(gamma=0.1, period=5.0,
                               period_bounds=(None, 10.0))
    b = k.get_bounds()
    assert b[0] == (None, None)
    assert b[1][0] is None
    assert np.allclose(b[1][1], np.log(10))

    # Test invalid bounds specification.
    try:
        kernels.ExpSine2Kernel(gamma=0.1, period=5.0, period_bounds=[10.0])
    except ValueError:
        pass
    else:
        assert False, "invalid bounds should fail"
