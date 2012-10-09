import numpy as np

from ._gp import _gp


class GaussianProcess(object):
    def __init__(self, pars):
        self.pars = pars
        self.gp = _gp(pars, 1)

    def __call__(self, x, y, x0, yerr=None, normalize=True):
        X = np.atleast_2d(x).T

        assert X.shape[1] + 1 == len(self.pars)

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

        self.gp.fit((X - xmean[None, :]) / xstd[None, :],
                    (y - ymean) / ystd,
                    yerr / ystd)
        mu, var = self.gp.predict((x0 - xmean[None, :]) / xstd[None, :])

        return ystd * mu + ymean, ystd * ystd * var, self.gp.evaluate()
