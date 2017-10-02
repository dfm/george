# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import warnings
import numpy as np
from scipy.linalg import LinAlgError

from . import kernels
from .solvers import TrivialSolver, BasicSolver
from .modeling import ModelSet, ConstantModel
from .utils import multivariate_gaussian_samples, nd_sort_samples


# MAGIC: tiny epsilon to add on the diagonal of the matrices in the absence
# of observational uncertainties. Needed for computational stability.
TINY = 1.25e-12


class GP(ModelSet):
    """
    The basic Gaussian Process object.

    :param kernel:
        An instance of a subclass of :class:`kernels.Kernel`.

    :param fit_kernel: (optional)
        If ``True``, the parameters of the kernel will be included in all
        the relevant methods (:func:`get_parameter_vector`,
        :func:`grad_log_likelihood`, etc.). (default: ``True``)

    :param mean: (optional)
        A description of the mean function. See :py:attr:`mean` for more
        information. (default: ``0.0``)

    :param fit_mean: (optional)
        If ``True``, the parameters of the mean function will be included in
        all the relevant methods (:func:`get_parameter_vector`,
        :func:`grad_log_likelihood`, etc.). (default: ``False``)

    :param white_noise: (optional)
        A description of the logarithm of the white noise variance added to
        the diagonal of the covariance matrix. See :py:attr:`white_noise` for
        more information. (default: ``log(TINY)``)

    :param fit_white_noise: (optional)
        If ``True``, the parameters of :py:attr:`white_noise` will be included
        in all the relevant methods (:func:`get_parameter_vector`,
        :func:`grad_log_likelihood`, etc.). (default: ``False``)

    :param solver: (optional)
        The solver to use for linear algebra as documented in :ref:`solvers`.

    :param kwargs: (optional)
        Any additional arguments are passed directly to the solver's init
        function.

    """

    def __init__(self,
                 kernel=None,
                 fit_kernel=True,
                 mean=None,
                 fit_mean=None,
                 white_noise=None,
                 fit_white_noise=None,
                 solver=None,
                 **kwargs):
        self._computed = False
        self._alpha = None
        self._y = None

        super(GP, self).__init__([
            ("mean",
             ConstantModel(0.0) if mean is None else _parse_model(mean)),
            ("white_noise", ConstantModel(np.log(TINY))
             if white_noise is None else _parse_model(white_noise)),
            ("kernel", kernels.EmptyKernel() if kernel is None else kernel),
        ])

        # Default behavior for fit_mean and fit_white_noise. If a constant is
        # provided and the fit_* parameter is not, we shouldn't try to fit
        # that component.
        try:
            float(mean)
        except TypeError:
            pass
        else:
            fit_mean = False if fit_mean is None else fit_mean

        try:
            float(white_noise)
        except TypeError:
            pass
        else:
            fit_white_noise = (
                False if fit_white_noise is None else fit_white_noise
            )

        if not fit_kernel:
            self.models["kernel"].freeze_all_parameters()
        if mean is None or (fit_mean is not None and not fit_mean):
            self.models["mean"].freeze_all_parameters()
        if white_noise is None or (fit_white_noise is not None and
                                   not fit_white_noise):
            self.models["white_noise"].freeze_all_parameters()

        if solver is None:
            trivial = (
                kernel is None or
                kernel.kernel_type == kernels.EmptyKernel.kernel_type
            )
            solver = TrivialSolver if trivial else BasicSolver
        self.solver_type = solver
        self.solver_kwargs = kwargs
        self.solver = None

    @property
    def mean(self):
        """
        An object (following the modeling protocol) that specifies the mean
        function of the GP. You can safely set this to a scalar, a callable,
        or an instance of a class satisfying the modeling protocol. In each
        case, the mean will be evaluated (either by calling the function or
        evaluating the :func:`get_value` method) at the input coordinates and
        it should return the one-dimensional mean evaluated at these
        coordinates.

        """
        return self.models["mean"]

    def _call_mean(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            mu = self.mean.get_value(x[:, 0]).flatten()
        else:
            mu = self.mean.get_value(x).flatten()
        if not np.all(np.isfinite(mu)):
            raise ValueError("mean function returned NaN or Inf for "
                             "parameters:\n{0}".format(
                                 self.mean.get_parameter_dict(
                                     include_frozen=True)))
        return mu

    def _call_mean_gradient(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            mu = self.mean.get_gradient(x[:, 0])
        else:
            mu = self.mean.get_gradient(x)
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            raise ValueError("mean gradient function returned NaN or Inf for "
                             "parameters:\n{0}".format(
                                 self.mean.get_parameter_dict(
                                     include_frozen=True)))
        return mu

    @property
    def white_noise(self):
        """
        An object (following the modeling protocol) that specifies the
        natural logarithm of the white noise variance added to the diagonal of
        the covariance matrix. You can safely set this to a scalar, a callable,
        or an instance of a class satisfying the modeling protocol. In each
        case, it will be evaluated (either by calling the function or
        evaluating the :func:`get_value` method) at the input coordinates and
        it should return the one-dimensional log-variance evaluated at these
        coordinates.

        This functionality is preferred to the ``WhiteKernel`` class provided
        by earlier versions of this module.

        """
        return self.models["white_noise"]

    def _call_white_noise(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            return self.white_noise.get_value(x[:, 0]).flatten()
        return self.white_noise.get_value(x).flatten()

    def _call_white_noise_gradient(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            return self.white_noise.get_gradient(x[:, 0])
        return self.white_noise.get_gradient(x)

    @property
    def computed(self):
        """
        Has the processes been computed since the last update of the kernel?

        """
        return (
            self._computed and
            self.solver.computed and
            (self.kernel is None or not self.kernel.dirty)
        )

    @computed.setter
    def computed(self, v):
        self._computed = v
        if v and self.kernel is not None:
            self.kernel.dirty = False

    def parse_samples(self, t):
        """
        Parse a list of samples to make sure that it has the correct
        dimensions.

        :param t: ``(nsamples,)`` or ``(nsamples, ndim)``
            The list of samples. If 1-D, this is assumed to be a list of
            one-dimensional samples otherwise, the size of the second
            dimension is assumed to be the dimension of the input space.

        Raises:
            ValueError: If the input dimension doesn't match the dimension of
                the kernel.

        """
        t = np.atleast_1d(t)
        # Deal with one-dimensional data.
        if len(t.shape) == 1:
            t = np.atleast_2d(t).T

        # Double check the dimensions against the kernel.
        if len(t.shape) != 2 or (self.kernel is not None and
                                 t.shape[1] != self.kernel.ndim):
            raise ValueError("Dimension mismatch")

        return t

    def _check_dimensions(self, y, check_dim=True):
        n, ndim = self._x.shape
        y = np.atleast_1d(y)
        if check_dim and len(y.shape) > 1:
            raise ValueError("The predicted dimension must be 1-D")
        if len(y) != n:
            raise ValueError("Dimension mismatch")
        return y

    def _compute_alpha(self, y, cache):
        # Recalculate alpha only if y is not the same as the previous y.
        if not cache:
            r = np.ascontiguousarray(self._check_dimensions(y) -
                                     self._call_mean(self._x),
                                     dtype=np.float64)
            return self.solver.apply_inverse(r, in_place=True).flatten()
        if self._alpha is None or not np.array_equiv(y, self._y):
            self._y = y
            r = np.ascontiguousarray(self._check_dimensions(y) -
                                     self._call_mean(self._x),
                                     dtype=np.float64)
            self._alpha = self.solver.apply_inverse(r, in_place=True).flatten()
        return self._alpha

    def apply_inverse(self, y):
        """
        Self-consistently apply the inverse of the computed kernel matrix to
        some vector or matrix of samples. This method subtracts the mean,
        sorts the samples, then returns the samples in the correct (unsorted)
        order.

        :param y: ``(nsamples, )`` or ``(nsamples, K)``
            The vector (or matrix) of sample values.

        """
        self.recompute(quiet=False)
        r = np.array(y, dtype=np.float64, order="F")
        r = self._check_dimensions(r, check_dim=False)

        # Broadcast the mean function
        m = [slice(None)] + [np.newaxis for _ in range(len(r.shape) - 1)]
        r -= self._call_mean(self._x)[m]

        # Do the solve
        if len(r.shape) == 1:
            b = self.solver.apply_inverse(r, in_place=True).flatten()
        else:
            b = self.solver.apply_inverse(r, in_place=True)
        return b

    def compute(self, x, yerr=0.0, **kwargs):
        """
        Pre-compute the covariance matrix and factorize it for a set of times
        and uncertainties.

        :param x: ``(nsamples,)`` or ``(nsamples, ndim)``
            The independent coordinates of the data points.

        :param yerr: (optional) ``(nsamples,)`` or scalar
            The Gaussian uncertainties on the data points at coordinates
            ``x``. These values will be added in quadrature to the diagonal of
            the covariance matrix.

        """
        # Parse the input coordinates and ensure the right memory layout.
        self._x = self.parse_samples(x)
        self._x = np.ascontiguousarray(self._x, dtype=np.float64)
        try:
            self._yerr2 = float(yerr)**2 * np.ones(len(x))
        except TypeError:
            self._yerr2 = self._check_dimensions(yerr) ** 2
        self._yerr2 = np.ascontiguousarray(self._yerr2, dtype=np.float64)

        # Set up and pre-compute the solver.
        self.solver = self.solver_type(self.kernel, **(self.solver_kwargs))

        # Include the white noise term.
        yerr = np.sqrt(self._yerr2 + np.exp(self._call_white_noise(self._x)))
        self.solver.compute(self._x, yerr, **kwargs)

        self._const = -0.5 * (len(self._x) * np.log(2 * np.pi) +
                              self.solver.log_determinant)
        self.computed = True
        self._alpha = None

    def recompute(self, quiet=False, **kwargs):
        """
        Re-compute a previously computed model. You might want to do this if
        the kernel parameters change and the kernel is labeled as ``dirty``.

        :param quiet: (optional)
            If ``True``, return false when the computation fails. Otherwise,
            throw an error if something goes wrong. (default: ``False``)

        """
        if not self.computed:
            if not (hasattr(self, "_x") and hasattr(self, "_yerr2")):
                raise RuntimeError("You need to compute the model first")
            try:
                # Update the model making sure that we store the original
                # ordering of the points.
                self.compute(self._x, np.sqrt(self._yerr2), **kwargs)
            except (ValueError, LinAlgError):
                if quiet:
                    return False
                raise
        return True

    def lnlikelihood(self, y, quiet=False):
        warnings.warn("'lnlikelihood' is deprecated. Use 'log_likelihood'",
                      DeprecationWarning)
        return self.log_likelihood(y, quiet=quiet)

    def log_likelihood(self, y, quiet=False):
        """
        Compute the logarithm of the marginalized likelihood of a set of
        observations under the Gaussian process model. You must call
        :func:`GP.compute` before this function.

        :param y: ``(nsamples, )``
            The observations at the coordinates provided in the ``compute``
            step.

        :param quiet:
            If ``True`` return negative infinity instead of raising an
            exception when there is an invalid kernel or linear algebra
            failure. (default: ``False``)

        """
        if not self.recompute(quiet=quiet):
            return -np.inf
        try:
            mu = self._call_mean(self._x)
        except ValueError:
            if quiet:
                return -np.inf
            raise
        r = np.ascontiguousarray(self._check_dimensions(y) - mu,
                                 dtype=np.float64)
        ll = self._const - 0.5 * self.solver.dot_solve(r)
        return ll if np.isfinite(ll) else -np.inf

    def grad_lnlikelihood(self, y, quiet=False):
        warnings.warn("'grad_lnlikelihood' is deprecated. "
                      "Use 'grad_log_likelihood'",
                      DeprecationWarning)
        return self.grad_log_likelihood(y, quiet=quiet)

    def grad_log_likelihood(self, y, quiet=False):
        """
        Compute the gradient of :func:`GP.log_likelihood` as a function of the
        parameters returned by :func:`GP.get_parameter_vector`. You must call
        :func:`GP.compute` before this function.

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
            return np.zeros(len(self), dtype=np.float64)

        # Pre-compute some factors.
        try:
            alpha = self._compute_alpha(y, False)
        except ValueError:
            if quiet:
                return np.zeros(len(self), dtype=np.float64)
            raise

        if len(self.white_noise) or len(self.kernel):
            K_inv = self.solver.get_inverse()
            A = np.einsum("i,j", alpha, alpha) - K_inv

        # Compute each component of the gradient.
        grad = np.empty(len(self))
        n = 0

        l = len(self.mean)
        if l:
            try:
                mu = self._call_mean_gradient(self._x)
            except ValueError:
                if quiet:
                    return np.zeros(len(self), dtype=np.float64)
                raise
            grad[n:n+l] = np.dot(mu, alpha)
            n += l

        l = len(self.white_noise)
        if l:
            wn = self._call_white_noise(self._x)
            wng = self._call_white_noise_gradient(self._x)
            grad[n:n+l] = 0.5 * np.sum((np.exp(wn)*np.diag(A))[None, :]*wng,
                                       axis=1)
            n += l

        l = len(self.kernel)
        if l:
            Kg = self.kernel.get_gradient(self._x)
            grad[n:n+l] = 0.5 * np.einsum("ijk,ij", Kg, A)

        return grad

    def nll(self, vector, y, quiet=True):
        self.set_parameter_vector(vector)
        if not np.isfinite(self.log_prior()):
            return np.inf
        return -self.log_likelihood(y, quiet=quiet)

    def grad_nll(self, vector, y, quiet=True):
        self.set_parameter_vector(vector)
        if not np.isfinite(self.log_prior()):
            return np.zeros(len(vector))
        return -self.grad_log_likelihood(y, quiet=quiet)

    def predict(self, y, t,
                return_cov=True,
                return_var=False,
                cache=True,
                kernel=None):
        """
        Compute the conditional predictive distribution of the model. You must
        call :func:`GP.compute` before this function.

        :param y: ``(nsamples,)``
            The observations to condition the model on.

        :param t: ``(ntest,)`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param return_cov: (optional)
            If ``True``, the full covariance matrix is computed and returned.
            Otherwise, only the mean prediction is computed. (default:
            ``True``)

        :param return_var: (optional)
            If ``True``, only return the diagonal of the predictive covariance;
            this will be faster to compute than the full covariance matrix.
            This overrides ``return_cov`` so, if both are set to ``True``,
            only the diagonal is computed. (default: ``False``)

        :param cache: (optional)
            If ``True`` the value of alpha will be cached to speed up repeated
            predictions.

        :param kernel: (optional)
            If provided, this kernel will be used to calculate the cross terms.
            This can be used to separate the predictions from different
            kernels.

        Returns ``mu``, ``(mu, cov)``, or ``(mu, var)`` depending on the values
        of ``return_cov`` and ``return_var``. These output values are:

        * **mu** ``(ntest,)``: mean of the predictive distribution,
        * **cov** ``(ntest, ntest)``: the predictive covariance matrix, and
        * **var** ``(ntest,)``: the diagonal elements of ``cov``.

        """
        self.recompute()
        alpha = self._compute_alpha(y, cache)
        xs = self.parse_samples(t)

        if kernel is None:
            kernel = self.kernel

        # Compute the predictive mean.
        Kxs = kernel.get_value(xs, self._x)
        mu = np.dot(Kxs, alpha) + self._call_mean(xs)
        if not (return_var or return_cov):
            return mu

        KinvKxs = self.solver.apply_inverse(Kxs.T)
        if return_var:
            var = kernel.get_value(xs, diag=True)
            var -= np.sum(Kxs.T*KinvKxs, axis=0)
            return mu, var

        cov = kernel.get_value(xs)
        cov -= np.dot(Kxs, KinvKxs)
        return mu, cov

    def sample_conditional(self, y, t, size=1):
        """
        Draw samples from the predictive conditional distribution. You must
        call :func:`GP.compute` before this function.

        :param y: ``(nsamples, )``
            The observations to condition the model on.

        :param t: ``(ntest, )`` or ``(ntest, ndim)``
            The coordinates where the predictive distribution should be
            computed.

        :param size: (optional)
            The number of samples to draw. (default: ``1``)

        Returns **samples** ``(size, ntest)``, a list of predictions at
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
            results = self.solver.apply_sqrt(np.random.randn(size, n))
            results += self._call_mean(self._x)
            return results[0] if size == 1 else results

        x = self.parse_samples(t)
        cov = self.get_matrix(x)
        cov[np.diag_indices_from(cov)] += TINY
        return multivariate_gaussian_samples(cov, size,
                                             mean=self._call_mean(x))

    def get_matrix(self, x1, x2=None):
        """
        Get the covariance matrix at a given set or two of independent
        coordinates.

        :param x1: ``(nsamples,)`` or ``(nsamples, ndim)``
            A list of samples.

        :param x2: ``(nsamples,)`` or ``(nsamples, ndim)`` (optional)
            A second list of samples. If this is given, the cross covariance
            matrix is computed. Otherwise, the auto-covariance is evaluated.

        """
        x1 = self.parse_samples(x1)
        if x2 is None:
            return self.kernel.get_value(x1)
        x2 = self.parse_samples(x2)
        return self.kernel.get_value(x1, x2)

    def get_value(self, *args, **kwargs):
        """
        A synonym for :func:`GP.log_likelihood` provided for consistency with
        the modeling protocol.

        """
        return self.log_likelihood(*args, **kwargs)

    def get_gradient(self, *args, **kwargs):
        """
        A synonym for :func:`GP.grad_log_likelihood` provided for consistency
        with the modeling protocol.

        """
        return self.grad_log_likelihood(*args, **kwargs)


def _parse_model(model):
    try:
        val = float(model)
    except TypeError:
        return model
    return ConstantModel(float(val))
