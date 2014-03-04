#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import time
import kplr
import emcee
import fitsio
import transit
import hashlib
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

d = os.path.dirname
sys.path.insert(0, d(d(d(d(os.path.abspath(__file__))))))
import george
from george.kernels import ExpSquaredKernel


class Model(object):

    def __init__(self, t, f, fe, texp, tol, mdep):
        self.t = t - np.min(t)
        self.period = 10 * np.max(self.t)
        self.texp, self.tol, self.mdep = texp, tol, mdep

        # Sample some parameters.
        self.q1, self.q2 = np.random.rand(), np.random.rand()
        self.t0 = np.max(self.t) * np.random.rand()
        self.tau = 0.05 + (1-0.05) * np.random.rand()
        self.ror = 0.008 + (0.2-0.008) * np.random.rand()
        self.b = np.random.rand()

        # Generate a synthetic transit.
        u1, u2 = 2*self.q1*self.q2, self.q1*(1-2*self.q2)
        lc = transit.ldlc_simple(self.t, u1, u2, self.period, self.t0,
                                 self.tau, self.ror, self.b, texp, tol, mdep)

        # Inject the transit and normalize the data.
        mu = np.median(f * lc)
        self.f = f * lc / mu
        self.fe = fe / mu

        # Set up the results directory.
        bp = self.basepath
        try:
            os.makedirs(bp)
        except os.error:
            pass

        pl.plot(self.t, self.f, ".k", ms=4, alpha=0.6)
        pl.savefig(os.path.join(bp, "data.png"))

    def _id(self):
        return hashlib.md5(str(hash((self.q1, self.q2, self.t0, self.tau,
                                     self.ror, self.b)))).hexdigest()

    @property
    def basepath(self):
        return os.path.join("results", self._id())

    def save(self):
        p = self.basepath

        # Save the full model.
        pickle.dump(self, open(os.path.join(p, "model.pkl"), "wb"), -1)

        # And the summary.
        open(os.path.join(p, "info.txt"), "w").write("""{0}
q1  = {1.q1}
q2  = {1.q2}
t0  = {1.t0}
tau = {1.tau}
ror = {1.ror}
b   = {1.b}
        """.format(self._id(), self))

    def lnlike(self, lna, lns, q1, q2, t0, tau, ror, b):
        kernel = ExpSquaredKernel(np.exp(lna), np.exp(lns))
        gp = george.GaussianProcess(kernel)
        gp.compute(self.t, self.fe)
        u1, u2 = 2*q1*q2, q1*(1-2*q2)
        lc = transit.ldlc_simple(self.t, u1, u2, self.period, t0, tau, ror, b,
                                 self.texp, self.tol, self.mdep)
        return gp.lnlikelihood(self.f - lc)

    def lnprior(self, lna, lns, q1, q2, t0, tau, ror, b):
        if not (-10 < lna < 0 and -2 < lns < 2):
            return -np.inf
        if not (0 < q1 < 1 and 0 < q2 < 1):
            return -np.inf
        if not (0 < t0 < 10 and 0 < tau < 1 and 0 < ror < 1 and 0 < b < 2):
            return -np.inf
        return 0.0

    def __call__(self, p):
        lp = self.lnprior(*p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(*p)

    def sample(self):
        # Initialize and compute initial probability.
        columns = [r"\alpha", "s", "q_1", "q_2", "t_0", r"\tau", "r/R", "b"]
        p0 = np.array([np.log(1e-2), np.log(2.0), self.q1, self.q2, self.t0,
                       self.tau, self.ror, self.b])
        s = time.time()
        lp = self(p0)
        s = time.time() - s
        print("initial ln-probability: {0}".format(lp))
        print("took {0} seconds".format(s))

        # Set up the sampler.
        ndim, nwalkers = len(p0), 32
        pos = [p0 + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self)

        # Run the sampling.
        print("sampling...")
        s = time.time()
        sampler.run_mcmc(pos, 10000)
        print("took {0} seconds".format(time.time() - s))

        # Save the results.
        pickle.dump((sampler.chain, sampler.lnprobability,
                     sampler.acceptance_fraction),
                    open(os.path.join(self.basepath, "results.pkl"), "wb"), -1)

        # Plot the results.
        print("plotting...")
        s = time.time()
        factor = 1.5
        bdim = 0.5 * factor   # size of bottom margin
        tdim = 0.05 * factor  # size of top margin
        fig, axes = pl.subplots(ndim+1, 1, figsize=(8, factor*(ndim+1)),
                                sharex=True)
        fig.subplots_adjust(left=0.17, bottom=bdim/(factor*ndim), right=0.9,
                            top=1-tdim/(factor*ndim), wspace=0.05, hspace=0.05)
        for i, (ax, c) in enumerate(zip(axes[:-1], columns)):
            ax.plot(sampler.chain[:, :, i].T, color="k", alpha=0.3)
            ax.set_ylabel("${0}$".format(c))
            ax.yaxis.set_major_locator(MaxNLocator(4))
        axes[-1].plot(sampler.lnprobability.T, color="k", alpha=0.3)
        axes[-1].yaxis.set_major_locator(MaxNLocator(4))
        axes[-1].set_xlim(0, sampler.chain.shape[1])
        axes[-1].set_xlabel("steps")
        axes[-1].set_ylabel(r"$\ln p$")
        fig.savefig(os.path.join(self.basepath, "time.png"))
        print("took {0} seconds".format(time.time() - s))


if __name__ == "__main__":
    # Load the data and mask missing data.
    data = fitsio.read("kplr004265377-2011177032512_llc.fits")
    t, f, fe, q = [data[k] for k in ["TIME", "SAP_FLUX", "SAP_FLUX_ERR",
                                     "SAP_QUALITY"]]
    m = (846 < t) * (t < 856) * np.isfinite(f) * np.isfinite(fe) * (q == 0)
    t, f, fe = t[m], f[m], fe[m]

    # Build the model.
    model = Model(t, f, fe, kplr.EXPOSURE_TIMES[1], 0.1, 2)
    print(model.basepath)
    model.save()
    model.sample()
