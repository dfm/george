import numpy as np
import scipy.optimize as op

from ._gp import _gp


class GaussianProcess(object):
    def __init__(self, pars):
        self.pars = pars
        self.gp = _gp(pars, 2)
        self.computed = False

    def prepare(self, x, y, yerr=None, normalize=True):
        # Make sure that the shape of the samples is ``(nsamples, ndim)``.
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.atleast_2d(x).T

        assert x.shape[1] + 1 == len(self.pars), "You need D+1 parameters " \
                                                 + "for the kernel."

        # Include uncertainties.
        if yerr is None:
            yerr = np.zeros_like(y)

        # Normalize the data.
        if normalize:
            self.xmean = np.mean(x, axis=0)
            self.xstd = np.std(x, axis=0)
            self.ymean = np.mean(y)
            self.ystd = np.std(y)
        else:
            self.xmean, self.xstd = np.zeros(x.shape[1]), np.ones(x.shape[1])
            self.ymean, self.ystd = 0.0, 1.0

        return ((x - self.xmean[None, :]) / self.xstd[None, :],
                (y - self.ymean) / self.ystd,
                yerr / self.ystd)

    def fit(self, *args, **kwargs):
        self.computed = False

        self.gp.fit(*(self.prepare(*args, **kwargs)))
        self.computed = True

        return args

    def predict(self, x0):
        assert self.computed
        r = self.gp.predict((x0 - self.xmean[None, :]) / self.xstd[None, :])
        return self.ystd * r[0] + self.ymean, self.ystd * self.ystd * r[1]

    def evaluate(self):
        assert self.computed
        return self.gp.evaluate()

    def optimize(self, *args, **kwargs):
        x, y, yerr = self.prepare(*args, **kwargs)

        def nll(p):
            gp = _gp(p ** 2, 2)
            gp.fit(x, y, yerr)
            r = -gp.evaluate()
            return r

        p = op.fmin_bfgs(nll, np.sqrt(self.pars), disp=False)
        self.pars = p ** 2
        self.gp = _gp(self.pars, 2)
        self.fit(*args, **kwargs)
