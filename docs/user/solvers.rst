.. module:: george

.. _solvers:

Solvers
=======

There are currently two different GP solvers included with George using
different libraries for doing linear algebra.
Both of the solvers implement the same API and should (up to some tolerance)
give the same answers on the same datasets.
The first solver :class:`GP` is a basic implementation using NumPy and SciPy
so it will be linked to whichever LAPACK/BLAS is used by those libraries.
This solver is very robust and it should be fairly efficient on small to
moderate datasets.
For larger datasets (:math:`N \gtrsim 5000`), you might consider using the
:class:`HODLRGP` solver.
This solver is built using Sivaram Amambikasaran's `HODLR library
<https://github.com/sivaramambikasaran/HODLR>`_ that implements
:math:`\mathcal{O}(N\,\log^2 N)` direct solves of dense matrices as described
`here <http://arxiv.org/abs/1403.6015>`_.


.. Basic Solver
.. ------------

.. .. autoclass:: george.GP
..    :inherited-members:


.. HODLR Solver
.. ------------

.. .. autoclass:: george.HODLRGP
..    :inherited-members:
