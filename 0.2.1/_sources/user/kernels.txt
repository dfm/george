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

.. _implementation:

Implementation Details
----------------------

It's worth understanding how these kernels are implemented.
Most of the hard work is done at a low level (in C++) and the Python is only a
thin wrapper to this functionality.
This makes the code fast and consistent across interfaces but it also means
that it isn't currently possible to implement new kernel functions efficiently
without recompiling the code.
Almost every kernel has hyperparameters that you can set to control its
behavior and these can be accessed via the ``pars`` property.
The values in this array are in the same order as you specified them when
initializing the kernel and, in the case of composite kernels (see
:ref:`combining-kernels`) the order goes from left to right.
For example,

.. code-block:: python

    from george import kernels

    k = 2.0 * kernels.Matern32Kernel(5.0)
    print(k.pars)
    # array([ 2.,  5.])


In general, kernel functions have some—possibly different—natural
parameterization that can be useful for parameter inference.
This can be accessed via the ``vector`` property and for most kernels, this
will be—unless otherwise specified—the natural logarithm of the ``pars``
array.
So, for our previous example,

.. code-block:: python

    k = 2.0 * kernels.Matern32Kernel(5.0)
    print(k.vector)
    # array([ 0.69314718,  1.60943791])

George is smart about when it recomputes the kernel and it will only do this
if you change the parameters.
Therefore, the best way to make changes is by *subscripting* the kernel.
It's worth noting that subscripting changes the ``vector`` array (not
``pars``) so following up our previous example, we can do something like

.. code-block:: python

    import numpy as np
    k = 2.0 * kernels.Matern32Kernel(5.0)

    k[0] = np.log(4.0)
    print(k.pars)
    # array([ 4.,  5.])

    k[:] = np.log([6.0, 10.0])
    print(k.pars)
    # array([ 6.,  10.])

.. note:: The gradient of each kernel is given with respect to ``vector`` not
    ``pars``. This means that in most cases the gradient taken in terms of the
    *logarithm* of the hyperparameters.



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

Implementing custom kernels in George is a bit of a pain in the current
version. For now, the only way to do it is with the :class:`PythonKernel`
where you provide a Python function that computes the value of the kernel
function at *a single pair of training points*.

.. autoclass:: george.kernels.PythonKernel
    :members:
