#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["MedianModel", "GPModel"]

import kplr
import transit
import numpy as np

import george
from george import kernels

from load_data import load_data

texp = kplr.EXPOSURE_TIMES[1] / 86400.0  # Long cadence exposure time.


class MedianModel(object):

    def __init__(self, fn, median=True):
        self.t, self.f, self.fe, self.truth = load_data(fn, median)
        self.ivar = 1.0 / self.fe ** 2
        self.central = transit.Central(q1=self.truth["q1"],
                                       q2=self.truth["q2"])
        self.system = transit.System(self.central)
        self.body = transit.Body(period=self.truth["period"],
                                 r=self.truth["r"],
                                 b=self.truth["b"],
                                 t0=self.truth["t0"])
        self.system.add_body(self.body)

    @property
    def true_vector(self):
        return np.array([
            1.0, self.truth["q1"], self.truth["q2"],
            np.log(self.truth["period"]), self.truth["t0"], self.truth["b"],
            np.log(self.truth["r"]),
        ])

    @property
    def vector(self):
        return np.array([
            self.central.flux, self.central.q1, self.central.q2,
            np.log(self.body.period), self.body.t0, self.body.b,
            np.log(self.body.r),
        ])

    @vector.setter
    def vector(self, v):
        self.central.flux, self.central.q1, self.central.q2 = v[:3]
        lnp, self.body.t0, self.body.b, lnr = v[3:]
        self.body.period = np.exp(lnp)
        self.body.r = np.exp(lnr)

    @property
    def labels(self):
        return [r"$f_\star$", r"$q_1$", r"$q_2$", r"$\ln P$",
                r"$t_0$", r"$b$", r"$\ln r$"]

    def lnlike(self, p):
        self.vector = p
        model = self.system.light_curve(self.t, texp=texp)
        return -0.5 * np.sum((self.f - model) ** 2 * self.ivar)

    def predict(self, t):
        return self.system.light_curve(t, texp=texp)

    def __call__(self, p):
        try:
            self.vector = p
            if np.log(self.body.r) < -4:
                return -np.inf
            if not (-0.5 < self.body.t0 < 0.5):
                return -np.inf
            if not self.body.b < 1.5:
                return -np.inf
            return self.lnlike(p)
        except ValueError:
            return -np.inf


class _mean_function(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, t):
        return self.s.light_curve(t, texp=texp)


class GPModel(MedianModel):

    def __init__(self, *args, **kwargs):
        kwargs["median"] = False
        super(GPModel, self).__init__(*args, **kwargs)

        # Normalize the fluxes.
        mu = np.median(self.f)
        self.f /= mu
        self.fe /= mu

        # Set up the GP model.
        self.kernel = 1e-6 * kernels.Matern32Kernel(3.0)
        self.gp = george.GP(self.kernel, mean=_mean_function(self.system))
        self.gp.compute(self.t, self.fe)

    @property
    def true_vector(self):
        return np.append([-np.inf for i in range(len(self.kernel))],
                         super(GPModel, self).true_vector)

    @property
    def vector(self):
        return np.append(np.log(self.kernel.pars), super(GPModel, self).vector)

    @vector.setter
    def vector(self, v):
        self.kernel.pars = np.exp(v[:len(self.kernel)])
        MedianModel.vector.fset(self, v[len(self.kernel):])

    @property
    def labels(self):
        return (map(r"$\ln\theta_{{{0}}}$".format,
                    range(1, len(self.kernel)+1))
                + super(GPModel, self).labels)

    def lnlike(self, p):
        self.vector = p
        return self.gp.lnlikelihood(self.f, quiet=True)

    def predict(self, t):
        return self.gp.sample_conditional(self.f, t)
