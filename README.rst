George
======

Installation:

::

    python setup.py install -I/path/to/eigen3

or

::

    python setup.py build_ext --inplace -I/path/to/eigen3

This requires `numpy <http://numpy.scipy.org>`_ and `Eigen3
<http://eigen.tuxfamily.org/>`_. If you install Eigen using Homebrew on a
Mac, ``/path/to/eigen`` is probably ``/usr/local/include/eigen3``.
