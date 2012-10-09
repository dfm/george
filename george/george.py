import numpy as np

from ._gp import evaluate


class GaussianProcess(object):
    def __init__(self, pars):
        self.pars = pars

    def __call__(self, x, y, x0, yerr=None, normalize=True):
        X = np.atleast_2d(x).T

        if yerr is None:
            yerr = np.zeros_like(y)

        if normalize:
            xmean = np.mean(X, axis=0)
            xstd = np.std(X, axis=0)
            ymean = np.mean(y)
            ystd = np.std(y)
        else:
            xmean, xstd = np.zeros(X.shape[1]), np.ones(X.shape[1])
            ymean, ystd = 0.0, 1.0

        mu, var, logprob = evaluate((X - xmean[None, :]) / xstd[None, :],
                                    (y - ymean) / ystd,
                                    yerr / ystd,
                                    (x0 - xmean[None, :]) / xstd[None, :],
                                    self.pars, 1)

        return ystd * mu + ymean, ystd * ystd * var, logprob
