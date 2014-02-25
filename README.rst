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

You'll first need to install `Eigen 3 <http://eigen.tuxfamily.org/>`_. Then,
just run::

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
