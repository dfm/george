import numpy as np

from ._gp import evaluate


class GaussianProcess(object):
    def __init__(self, pars):
        self.pars = pars

    def __call__(self, x, y, x0, yerr=None):
        X = np.array(x)
        if len(X.shape) == 1:
            X = np.atleast_2d(X).T

        if yerr is None:
            yerr = np.zeros_like(y)

        xmean = np.mean(X, axis=0)
        xstd = np.std(X, axis=0)
        ymean = np.mean(y)
        ystd = np.std(y)

        mu, var, logprob = evaluate((X - xmean) / xstd,
                                    (y - ymean) / ystd,
                                    yerr / ystd,
                                    (x0 - xmean) / xstd,
                                    self.pars, 1)

        return ystd * mu + ymean, ystd * ystd * var, logprob
