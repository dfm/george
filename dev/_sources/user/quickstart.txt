.. _quickstart:

Getting started
===============

Installation
------------

You can install the most recent stable version of George using `PyPI
<#stable>`_ or the development version from `GitHub
<https://github.com/dfm/george>`_.

Prerequisites
+++++++++++++

Whichever method you choose, you'll need to make sure that you first have
`Eigen <http://eigen.tuxfamily.org/>`_ installed.
On Debian-based Linux distributions:

.. code-block:: bash

    sudo apt-get install libeigen3-dev

On Mac:

.. code-block:: bash

    brew install eigen

.. note:: Chances are high that George won't work on Windows right now because
    it hasn't been tested at all but feel free to try it out at your own risk!

You'll also need a working scientific Python installation (including `NumPy
<http://www.numpy.org/>`_ and `SciPy <http://www.scipy.org/>`_).
I recommend the `Anaconda distribution <http://continuum.io/downloads>`_ if
you don't already have your own opinions.

.. _stable:

Stable Version
++++++++++++++

The simplest way to install the `most recent stable version of George
<https://pypi.python.org/pypi/george>`_ is using `pip
<http://pip.readthedocs.org/>`_:

.. code-block:: bash

    pip install george

If you installed Eigen in a strange place, specify that location by running
(sorry to say that it's pretty freaking ugly):

.. code-block:: bash

    pip install george \
        --global-option=build_ext \
        --global-option=-I/path/to/eigen3


.. _dev:

Development Version
+++++++++++++++++++

To get the source for the development version, clone the git repository and
checkout the required HODLR submodule:

.. code-block:: bash

    git clone https://github.com/dfm/george.git
    cd george
    git submodule init
    git submodule update

Then, install the package by running the following command:

.. code-block:: bash

    python setup.py install

If installed Eigen in a non-standard location, you can specify the correct
path using the install command:

.. code-block:: bash

    python setup.py build_ext -I/path/to/eigen3 install

Testing
+++++++

To run the unit tests, install `nose <https://nose.readthedocs.org>`_ and then
execute:

.. code-block:: bash

    nosetests -v george.testing

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/george/issues>`_.


Examples
--------

Take a look at :ref:`first` to get started and then check out the other
tutorials for some more advanced usage examples.
