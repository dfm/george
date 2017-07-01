.. _quickstart:

Getting started
===============

Installation
------------

You can install the most recent stable version of George using `PyPI
<#stable>`_ or the development version from `GitHub
<https://github.com/dfm/george>`_.

You'll need a working scientific Python installation (including `NumPy
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


.. _dev:

Development Version
+++++++++++++++++++

To get the source for the development version, clone the git repository and
checkout the required HODLR submodule:

.. code-block:: bash

    git clone --recursive https://github.com/dfm/george.git
    cd george

Then, install the package by running the following command:

.. code-block:: bash

    python setup.py install

Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    py.test -v

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/george/issues>`_.

Examples
--------

Take a look at :ref:`first` to get started and then check out the other
tutorials for some more advanced usage examples.
