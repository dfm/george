# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
import scipy.optimize as op
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from .utils import multivariate_gaussian_samples, nd_sort_samples


# MAGIC: tiny epsilon to add on the diagonal of the matrices in the absence
# of observational uncertainties. Needed for computational stability.
TINY = 1.25e-12


class GP(object):
    """
    The basic Gaussian Process object.

    :param kernel:
        An instance of a subclass of :class:`kernels.Kernel`.

    :param mean: (optional)
        A description of the mean function; can be a callable or a scalar. If
        scalar, the mean is assumed constant. Otherwise, the function will be
        called with the array of independent coordinates as the only argument.
        (default: ``0.0``)

    """

    def __init__(self, kernel, mean=None):
        self.kernel = kernel
        self._computed = False
        if mean is None:
            self.mean = _default_mean(0.)
        else:
            try:
                val = float(mean)
            except TypeError:
                self.mean = mean
            else:
                self.mean = _default_mean(val)

    @property
    def computed(self):
        """
        Has the processes been computed since the last update of the kernel?

        """
        return self._computed and not self.kernel.dirty

    @computed.setter
    def computed(self, v):
        self._computed = v
        if v:
            self.kernel.dirty = False

    def parse_samples(self, t, sort=False):
        """
        Parse a list of samples to make sure that it has the correct
        dimensions and optionally sort it. In one dimension, the samples will
        be sorted in the logical order. In higher dimensions, a kd-tree is
        built and the samples are sorted in increasing distance from the
        *first* sample.

        :param t: ``(nsamples,)`` or ``(nsamples, ndim)``
            The list of samples. If 1-D, this is assumed to be a list of
            one-dimensional samples otherwise, the size of the second
            dimension is assumed to be the dimension of the input space.

        :param sort:
            A boolean flag indicating whether or not the samples should be
            sorted.

        Returns a tuple ``(samples, inds)`` where

        * **samples** is an array with shape ``(nsamples, ndim)`` and if
          ``sort`` was ``True``, it will also be sorted, and
        * **inds** is an ``(nsamples,)`` list of integer permutations used to
          sort the list of samples.

        Raises a ``RuntimeError`` if the input dimension doesn't match the
        dimension of the kernel.

        """
        t = np.atleast_1d(t)
        if len(t.shape) == 1:
            # Deal with one-dimensional data.
            if sort:
                inds = np.argsort(t)
            else:
                inds = np.arange(len(t), dtype=int)
            t = np.atleast_2d(t).T
        elif sort:
            # Sort the data using a KD-tree.
            inds = nd_sort_samples(t)
        else:
            # Otherwise, assume that the samples are sorted.
            inds = np.arange(t.shape[0], dtype=int)

        # Double check the dimensions against the kernel.
        if len(t.shape) != 2 or t.shape[1] != self.kernel.ndim:
            raise ValueError("Dimension mismatch")

        return t[inds], inds

    def _check_dimensions(self, y):
        n, ndim = self._x.shape
        y = np.atleast_1d(y)
        if len(y.shape) > 1:
            raise ValueError("The predicted dimension must be 1-D")
        if len(y) != n:
            raise ValueError("Dimension mismatch")
        return y

    def compute(self, x, yerr=TINY, sort=True, **kwargs):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :param x: ``(nsamples,)`` or ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param yerr: (optional) ``(nsamples,)`` or scalar
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        :param sort: (optional)
            Should the samples be sorted before computing the covariance
            matrix? This can lead to more numerically stable results and with
            some linear algebra libraries this can more computationally
            efficient. Either way, this flag is passed directly to
            :func:`parse_samples`. (default: ``True``)

        """
        # Parse the input coordinates.
        self._x, self.inds = self.parse_samples(x, sort)
        try:
            self._yerr = float(yerr) * np.ones(len(x))
        except TypeError:
            self._yerr = self._check_dimensions(yerr)[self.inds]
        self._do_compute(**kwargs)

    def _do_compute(self, _scale=0.5*np.log(2*np.pi)):
        # Compute the kernel matrix.
        K = self.kernel(self._x[:, None], self._x[None, :])
        K[np.diag_indices_from(K)] += self._yerr ** 2

        # Factor the matrix and compute the log-determinant.
        factor, _ = self._factor = cho_factor(K, overwrite_a=True)
        self._const = -(np.sum(np.log(np.diag(factor))) + _scale*len(self._x))

        # Save the computed state.
        self.computed = True

    def recompute(self, sort=False, **kwargs):
        """
        Re-compute a previously computed model. You might want to do this if
        the kernel parameters change and the kernel is labeled as ``dirty``.

        :params sort: (optional)
            Should the samples be sorted before computing the covariance
            matrix? (default: ``False``)

        """
        if not (hasattr(self, "_x") and hasattr(self, "_yerr")):
            raise RuntimeError("You need to compute the model first")
        return self.compute(self._x, self._yerr, sort=sort, **kwargs)

    def _compute_lnlike(self, r):
        return self._const - 0.5*np.dot(r.T, cho_solve(self._factor, r))

    def lnlikelihood(self, y, quiet=False):
        """
        Compute the ln-likelihood of a set of observations under the Gaussian
        process model. You must call ``compute`` before this function.

        :param y: ``(nsamples, )``
            The observations at the coordinates provided in the ``compute``
            step.

        :param quiet:
            If ``True`` return negative infinity instead of raising an
            exception when there is an invalid kernel or linear algebra
            failure. (default: ``False``)

        """
        if not self.computed:
            try:
                self.recompute()
            except (ValueError, LinAlgError):
                if quiet:
                    return -np.inf
                raise
        r = self._check_dimensions(y)[self.inds] - self.mean(self._x)
        ll = self._compute_lnlike(r)
        return ll if np.isfinite(ll) else -np.inf

    def grad_lnlikelihood(self, y, dims=None, quiet=False):
        """
        Compute the gradient of the ln-likelihood function as a function of
        the kernel parameters.

        :param y: ``(nsamples,)``
            The list of observations at coordinates ``x`` provided to the
            :func:`compute` function.

        :param dims: (optional)
            If you only want to compute the gradient in some dimensions,
            list them here.

        :param quiet:
            If ``True`` return a gradient of zero instead of raising an
            exception when there is an invalid kernel or linear algebra
            failure. (default: ``False``)

        """
        # By default, compute the gradient in all dimensions.
        if dims is None:
            dims = np.ones(len(self.kernel), dtype=bool)

        # Make sure that the model is computed and try to recompute it if it's
        # dirty.
        if not self.computed:
            try:
                self.recompute()
            except (ValueError, LinAlgError):
                if quiet:
                    return np.zeros_like(dims, dtype=float)
                raise

        # Parse the input sample list.
        r = self._check_dimensions(y)[self.inds] - self.mean(self._x)

        # Pre-compute some factors.
        alpha = cho_solve(self._factor, r)
        Kg = self.kernel.grad(self._x[:, None], self._x[None, :])[dims]

        # Loop over dimensions and compute the gradient in each one.
        g = np.empty(len(Kg))
        for i, k in enumerate(Kg):
            d = sum(map(lambda r: np.dot(alpha, r), alpha[:, None] * k))
            d -= np.sum(np.diag(cho_solve(self._factor, k)))
            g[i] = 0.5 * d

        return g

    def predict(self, y, t):
        """
        Compute the conditional predictive distribution of the model.

        :param y: ``(nsamples,)``
            The observations to condition the model on.

        :param t: ``(ntest,)`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        Returns a tuple ``(mu, cov)`` where

        * **mu** ``(ntest,)`` is the mean of the predictive distribution, and
        * **cov** ``(ntest, ntest)`` is the predictive covariance.

        """
        if not self.computed:
            self.recompute()

        r = self._check_dimensions(y)[self.inds] - self.mean(self._x)
        xs, i = self.parse_samples(t, False)
        alpha = cho_solve(self._factor, r)

        # Compute the predictive mean.
        Kxs = self.kernel(self._x[None, :], xs[:, None])
        mu = np.dot(Kxs, alpha) + self.mean(xs)

        # Compute the predictive covariance.
        cov = self.kernel(xs[:, None], xs[None, :])
        cov -= np.dot(Kxs, cho_solve(self._factor, Kxs.T))

        return mu, cov

    def sample_conditional(self, y, t, size=1):
        """
        Draw samples from the predictive conditional distribution.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param size: (optional)
            The number of samples to draw. (default: ``1``)

        Returns **samples** ``(N, ntest)``, a list of predictions at
        coordinates given by ``t``.

        """
        mu, cov = self.predict(y, t)
        return multivariate_gaussian_samples(cov, size, mean=mu)

    def sample(self, t, size=1):
        """
        Draw samples from the prior distribution.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the model should be sampled.

        :param size: (optional)
            The number of samples to draw. (default: ``1``)

        Returns **samples** ``(N, ntest)``, a list of predictions at
        coordinates given by ``t``.

        """
        x, _ = self.parse_samples(t, False)
        cov = self.get_matrix(x)
        return multivariate_gaussian_samples(cov, size, mean=self.mean(x))

    def get_matrix(self, t):
        """
        Get the covariance matrix at a given set of independent coordinates.

        :param t: ``(nsamples,)`` or ``(nsamples, ndim)``
            The list of samples.

        """
        r, _ = self.parse_samples(t, False)
        return self.kernel(r[:, None], r[None, :])

    def optimize(self, x, y, yerr=TINY, sort=True, dims=None, in_log=True,
                 verbose=True, **kwargs):
        """
        A simple and not terribly robust non-linear optimization algorithm for
        the kernel hyperpararmeters.

        :param x: ``(nsamples,)`` or ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param y: ``(nsamples, )``
            The observations at the coordinates ``x``.

        :param yerr: (optional) ``(nsamples,)`` or scalar
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        :param sort: (optional)
            Should the samples be sorted before computing the covariance
            matrix?

        :param dims: (optional)
            If you only want to optimize over some parameters, list their
            indices here.

        :param in_log: (optional) ``(len(kernel),)``, ``(len(dims),)`` or bool
            If you want to fit the parameters in the log (this can be useful
            for parameters that shouldn't go negative) specify that here. This
            can be a single boolean---in which case it is assumed to apply to
            every dimension---or it can be an array of booleans, one for each
            dimension.

        :param verbose: (optional)
            Display the results of the call to :func:`scipy.optimize.minimize`?
            (default: ``True``)

        Returns ``(pars, results)`` where ``pars`` is the list of optimized
        parameters and ``results`` is the results object returned by
        :func:`scipy.optimize.minimize`.

        """
        self.compute(x, yerr, sort=sort)

        # By default, optimize all the hyperparameters.
        if dims is None:
            dims = np.ones(len(self.kernel), dtype=bool)
        dims = np.arange(len(self.kernel))[dims]

        # Deal with conversion functions.
        try:
            len(in_log)
        except TypeError:
            in_log = in_log * np.ones_like(dims, dtype=bool)
        else:
            if len(in_log) != len(dims):
                raise RuntimeError("Dimension list and log mask mismatch")

        # Build the conversion functions.
        conv = np.array([lambda x: x for i in range(len(dims))])
        iconv = np.array([lambda x: x for i in range(len(dims))])
        conv[in_log] = np.exp
        iconv[in_log] = np.log

        # Define the objective function and gradient.
        def nll(pars):
            for i, f, p in izip(dims, conv, pars):
                self.kernel[i] = f(p)

            ll = self.lnlikelihood(y, quiet=True)
            if not np.isfinite(ll):
                return 1e25  # The optimizers can't deal with infinities.
            return -ll

        def grad_nll(pars):
            for i, f, p in izip(dims, conv, pars):
                self.kernel[i] = f(p)
            return -self.grad_lnlikelihood(y, dims=dims, quiet=True)

        # Run the optimization.
        p0 = [f(p) for f, p in izip(iconv, self.kernel.pars[dims])]
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)

        if verbose:
            print(results.message)

        # Update the kernel.
        for i, f, p in izip(dims, conv, results.x):
            self.kernel[i] = f(p)

        return self.kernel.pars[dims], results


class _default_mean(object):

    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value + np.zeros(len(t), dtype=float)
