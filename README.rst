George
======

Fast Gaussian processes for regression
--------------------------------------

This is a GP code built on top of Sivaram Ambikasaran's `HODLR
solver <https://github.com/sivaramambikasaran/HODLR_Solver>`_ designed to
be fast on **huge** problems. So far it's only being used to model the noise
in *Kepler* light curves but it should be more generally useful.

The code is mainly written in (undocumented) C++ with Python bindings.

Installation
------------

**Latest stable release**

You'll first need to install `Eigen 3 <http://eigen.tuxfamily.org/>`_ and
(obviously) `NumPy <http://www.numpy.org/>`_. Then, for the default
installation, just run::

  pip install george

If you have installed Eigen in a strange place, you'll need to install from
the git repository.

**Development version**

To install the development version, first clone the repository and initialize
the submodules::

  git clone https://github.com/dfm/george
  cd george
  git submodule init
  git submodule update

Then, install the package using::

  python setup.py install

If you've installed Eigen in a strange place, you can specify this by running::

  python setup.py install --eigen-include=/path/to/eigen

Usage
-----

Here's the simplest possible example of how you might want to use George::

  import numpy as np
  import george
  from george.kernels import ExpSquaredKernel
  
  # Generate some fake noisy data.
  x = 10 * np.sort(np.random.rand(10))
  yerr = 0.2 * np.ones_like(x)
  y = np.sin(x) + yerr * np.random.randn(len(x))
  
  # Set up the Gaussian process.
  kernel = ExpSquaredKernel(1.0, 1.0)
  gp = george.GaussianProcess(kernel)
  
  # Pre-compute the factorization of the matrix.
  gp.compute(x, yerr)
  
  # Compute the log likelihood.
  print(gp.lnlikelihood(y))
  
  # Draw 100 samples from the predictive conditional distribution.
  t = np.linspace(0, 10, 500)
  samples = gp.sample_conditional(y, t, size=100)

License
-------

George is being developed by `Dan Foreman-Mackey <http://dfm.io>`_ and the source
is available under the terms of the `MIT license
<https://github.com/dfm/george/blob/master/LICENSE>`_.

Copyright 2012-2014 Dan Foreman-Mackey
