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
.. autoclass:: george.kernels.ConstantKernel
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

*Coming soon*


.. _new-kernels:

Implementing New Kernels
------------------------

*Coming soon*
