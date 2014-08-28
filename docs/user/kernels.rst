.. module:: george.kernels

.. _kernels:

Kernels
=======

George comes equipped with a suite of standard covariance functions or
kernels that can be combined to build more complex models.
The standard kernels fall into the following categories:

1. :ref:`basic-kernels` — trivial (constant or parameterless) functions,
2. :ref:`radial-kernels` — functions that depend only on the radial distance
   between points in some user-defined metric, and
3. :ref:`periodic-kernels` — exactly period functions that, when combined with
   a radial kernel, can model quasi-periodic signals.

:ref:`combining-kernels` describes how to combine kernels to build more
sophisticated models and :ref:`new-kernels` explains how you would go about
incorporating a custom kernel.

**Note:** every kernel takes an optional ``ndim`` keyword that must be set to
the number of input dimensions for your problem.


.. _basic-kernels:

Basic Kernels
-------------

.. autoclass:: george.kernels.Kernel
    :special-members: __call__
    :members:

.. autoclass:: george.kernels.ConstantKernel
.. autoclass:: george.kernels.WhiteKernel
.. autoclass:: george.kernels.DotProductKernel


.. _radial-kernels:

Radial Kernels
--------------

.. autoclass:: george.kernels.RadialKernel
.. autoclass:: george.kernels.ExpKernel
.. autoclass:: george.kernels.ExpSquaredKernel
.. autoclass:: george.kernels.RBFKernel
.. autoclass:: george.kernels.Matern32Kernel
.. autoclass:: george.kernels.Matern52Kernel


.. _periodic-kernels:

Periodic Kernels
----------------

.. autoclass:: george.kernels.CosineKernel
.. autoclass:: george.kernels.ExpSine2Kernel


.. _combining-kernels:

Combining Kernels
-----------------

More complicated kernels can be constructed by algebraically combining the
basic kernels listed in the previous sections.
In particular, all the kernels support addition and multiplication.
For example, an exponential-squared kernel with a non-trivial variance can be
constructed as follows:

.. code-block:: python

    from george import kernels
    kernel = 1e-3 * kernels.ExpSquaredKernel(3.4)

This is equivalent to:

.. code-block:: python

    from math import sqrt
    kernel = kernels.Product(kernels.ConstantKernel(sqrt(1e-3)),
                             kernels.ExpSquaredKernel(3.4))

As demonstrated in :ref:`hyper`, a mixture of kernels can be implemented with
addition:

.. code-block:: python

    k1 = 1e-3 * kernels.ExpSquaredKernel(3.4)
    k2 = 1e-4 * kernels.Matern32Kernel(14.53)
    kernel = k1 + k2


.. _new-kernels:

Implementing New Kernels
------------------------

*Coming soon*
