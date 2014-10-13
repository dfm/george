.. module:: george

.. _solvers:

Solvers
=======

There are currently two different GP solvers included with George using
different libraries for doing linear algebra.
Both of the solvers implement the same API and should (up to some tolerance)
give the same answers on the same datasets.
A solver is just a class that takes a :class:`Kernel` and that exposes 3
methods:

1. ``compute`` --- to compute and factorize the kernel matrix,
2. ``apply_inverse`` --- to left-multiply the input by the covariance matrix
   :math:`C^{-1}\,b` (actually implemented by solving the system
   :math:`C\,x = b`), and
3. ``apply_sqrt`` --- to apply the (Cholesky) square root of the covariance.

The solvers also provide the properties ``computed`` and ``log_determinant``.

The simplest solver provided by George (:class:`BasicSolver`) uses `scipy's
Cholesky implementation
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html>`_
and the second implementation (:class:`HODLRSolver`) uses  Sivaram
Amambikasaran's `HODLR library <https://github.com/sivaramambikasaran/HODLR>`_.
The HODLR algorithm implements a :math:`\mathcal{O}(N\,\log^2 N)` direct
solver for dense matrices as described `here <http://arxiv.org/abs/1403.6015>`_.

By default, George uses the :class:`BasicSolver` but the :class:`HODLRSolver`
can be used as follows:

.. code-block:: python

    import george
    kernel = ...
    gp = george.GP(kernel, solver=george.HODLRSolver)

The :class:`HODLRSolver` is probably best for most one-dimensional problems
and some large multi-dimensional problems but it doesn't (in general) scale
well with the number of input dimensions.
In practice, it's worth trying both solvers on your specific problem to see
which runs faster.


Basic Solver
------------

.. autoclass:: george.BasicSolver
   :inherited-members:


HODLR Solver
------------

.. autoclass:: george.HODLRSolver
   :inherited-members:
