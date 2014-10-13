# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import numpy as np
import scipy.optimize as op
from scipy.linalg import LinAlgError

from .basic import BasicSolver
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

    :param solver: (optional)
        The solver to use for linear algebra as documented in :ref:`solvers`.

    :param kwargs: (optional)
        Any additional arguments are passed directly to the solver's init
        function.

    """

    def __init__(self, kernel, mean=None, solver=BasicSolver, **kwargs):
        self.kernel = kernel
        self._computed = False
        self._alpha = None
        self._y = None
        self.mean = mean

        self.solver_type = solver
        self.solver_kwargs = kwargs
        self.solver = None

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if mean is None:
            self._mean = _default_mean(0.)
        else:
            try:
                val = float(mean)
            except TypeError:
                self._mean = mean
            else:
                self._mean = _default_mean(val)

    @property
    def computed(self):
        """
        Has the processes been computed since the last update of the kernel?

        """
        return (
            self._computed
            and self.solver.computed
            and not self.kernel.dirty
        )

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

    def _compute_alpha(self, y):
        # Recalculate alpha only if y is not the same as the previous y.
        if self._alpha is None or not np.array_equiv(y, self._y):
            self._y = y
            r = np.ascontiguousarray(self._check_dimensions(y)[self.inds]
                                     - self.mean(self._x), dtype=np.float64)
            self._alpha = self.solver.apply_inverse(r, in_place=True)

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
        # Parse the input coordinates and ensure the right memory layout.
        self._x, self.inds = self.parse_samples(x, sort)
        self._x = np.ascontiguousarray(self._x, dtype=np.float64)
        try:
            self._yerr = float(yerr) * np.ones(len(x))
        except TypeError:
            self._yerr = self._check_dimensions(yerr)[self.inds]
        self._yerr = np.ascontiguousarray(self._yerr, dtype=np.float64)

        # Set up and pre-compute the solver.
        self.solver = self.solver_type(self.kernel, **(self.solver_kwargs))
        self.solver.compute(self._x, self._yerr, **kwargs)

        self._const = -0.5 * (len(self._x) * np.log(2 * np.pi)
                              + self.solver.log_determinant)
        self.computed = True

        self._alpha = None

    def recompute(self, quiet=False, **kwargs):
        """
        Re-compute a previously computed model. You might want to do this if
        the kernel parameters change and the kernel is labeled as ``dirty``.

        """
        if self.kernel.dirty or not self.computed:
            if not (hasattr(self, "_x") and hasattr(self, "_yerr")):
                raise RuntimeError("You need to compute the model first")
            try:
                # Update the model making sure that we store the original
                # ordering of the points.
                initial_order = np.array(self.inds)
                self.compute(self._x, self._yerr, sort=False, **kwargs)
                self.inds = initial_order
            except (ValueError, LinAlgError):
                if quiet:
                    return False
                raise
        return True

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
        r = np.ascontiguousarray(self._check_dimensions(y)[self.inds]
                                 - self.mean(self._x), dtype=np.float64)
        if not self.recompute(quiet=quiet):
            return -np.inf
        ll = self._const - 0.5 * np.dot(r, self.solver.apply_inverse(r))
        return ll if np.isfinite(ll) else -np.inf

    def grad_lnlikelihood(self, y, quiet=False):
        """
        Compute the gradient of the ln-likelihood function as a function of
        the kernel parameters.

        :param y: ``(nsamples,)``
            The list of observations at coordinates ``x`` provided to the
            :func:`compute` function.

        :param quiet:
            If ``True`` return a gradient of zero instead of raising an
            exception when there is an invalid kernel or linear algebra
            failure. (default: ``False``)

        """
        # Make sure that the model is computed and try to recompute it if it's
        # dirty.
        if not self.recompute(quiet=quiet):
            return np.zeros(len(self.kernel), dtype=float)

        # Pre-compute some factors.
        self._compute_alpha(y)
        K_inv = self.solver.apply_inverse(np.eye(self._alpha.size),
                                          in_place=True)
        Kg = self.kernel.gradient(self._x)

        # Calculate the gradient.
        A = np.outer(self._alpha, self._alpha) - K_inv
        g = 0.5 * np.einsum('ijk,ij', Kg, A)

        return g

    def predict(self, y, t, mean_only=False):
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
        self.recompute()
        self._compute_alpha(y)
        xs, i = self.parse_samples(t, False)

        # Compute the predictive mean.
        Kxs = self.kernel.value(xs, self._x)
        mu = np.dot(Kxs, self._alpha) + self.mean(xs)
        if mean_only:
            return mu

        # Compute the predictive covariance.
        KxsT = np.ascontiguousarray(Kxs.T, dtype=np.float64)
        cov = self.kernel.value(xs)
        cov -= np.dot(Kxs, self.solver.apply_inverse(KxsT, in_place=True))

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

    def sample(self, t=None, size=1):
        """
        Draw samples from the prior distribution.

        :param t: ``(ntest, )`` or ``(ntest, ndim)`` (optional)
            The coordinates where the model should be sampled. If no
            coordinates are given, the precomputed coordinates and
            factorization are used.

        :param size: (optional)
            The number of samples to draw. (default: ``1``)

        Returns **samples** ``(size, ntest)``, a list of predictions at
        coordinates given by ``t``. If ``size == 1``, the result is a single
        sample with shape ``(ntest,)``.

        """
        if t is None:
            self.recompute()
            n, _ = self._x.shape

            # Generate samples using the precomputed factorization.
            samples = self.solver.apply_sqrt(np.random.randn(size, n))
            samples += self.mean(self._x)

            # Reorder the samples correctly.
            results = np.empty_like(samples)
            results[:, self.inds] = samples
            return results[0] if size == 1 else results

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
        return self.kernel.value(r)

    def optimize(self, x, y, yerr=TINY, sort=True, dims=None, verbose=True,
                 **kwargs):
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

        # Define the objective function and gradient.
        def nll(pars):
            self.kernel[dims] = pars
            ll = self.lnlikelihood(y, quiet=True)
            if not np.isfinite(ll):
                return 1e25  # The optimizers can't deal with infinities.
            return -ll

        def grad_nll(pars):
            self.kernel[dims] = pars
            return -self.grad_lnlikelihood(y, quiet=True)[dims]

        # Run the optimization.
        p0 = self.kernel.vector[dims]
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)

        if verbose:
            print(results.message)

        return self.kernel.vector[dims], results


class _default_mean(object):

    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value + np.zeros(len(t), dtype=float)

    def __len__(self):
        return 1

    @property
    def vector(self):
        return np.array([self.value])

    @vector.setter
    def vector(self, value):
        self.value = float(value)

    def lnprior(self):
        return 0.0
