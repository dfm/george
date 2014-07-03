.. _kernels:
.. module:: george.kernels

Kernels
=======

There are a bunch of standard default kernels:

1. constant kernel (:class:`ConstantKernel`),
2. exponential-squared (:class:`ExpSquaredKernel`),
3. dude.


Basic Kernels
-------------

.. autoclass:: george.kernels.ConstantKernel
.. autoclass:: george.kernels.DotProductKernel


Radial Kernels
--------------

.. autoclass:: george.kernels.RadialKernel
.. autoclass:: george.kernels.ExpKernel
.. autoclass:: george.kernels.ExpSquaredKernel
.. autoclass:: george.kernels.RBFKernel
.. autoclass:: george.kernels.Matern32Kernel
.. autoclass:: george.kernels.Matern52Kernel


Periodic Kernels
----------------

.. autoclass:: george.kernels.CosineKernel
.. autoclass:: george.kernels.ExpSine2Kernel


Combining Kernels
-----------------
