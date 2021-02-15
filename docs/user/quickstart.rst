.. _quickstart:

Getting started
===============

Installation
------------

The core implementation of george is written in C++ so this will need to be
compiled to be called from Python. The easiest way for a new user to do this
will be by following the directions in the :ref:`using-conda` section below.

.. _using-conda:

Using conda
+++++++++++

The easiest way to install george is using `conda
<http://continuum.io/downloads>`_ (via `conda-forge
<https://conda-forge.github.io/>`_) with the following command:

.. code-block:: bash

    conda install -c conda-forge george


Using pip
+++++++++

George can also be installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install george

.. _source:

From Source
+++++++++++

The source code for george can be downloaded `from GitHub
<https://github.com/dfm/george>`_ by running

.. code-block:: bash

    git clone --recursive https://github.com/dfm/george.git


.. _python-deps:

**Dependencies**

You'll need a Python installation and I recommend `conda
<http://continuum.io/downloads>`_ if you don't already have your own opinions.

After installing Python, the following dependencies are required to build and
run george:

1. `NumPy <http://www.numpy.org/>`_,
2. `SciPy <http://www.numpy.org/>`_, and
3. `pybind11 <https://pybind11.readthedocs.io>`_.

If you're using conda, you can install all of the dependencies with the
following command:

.. code-block:: bash

    conda install -c conda-forge numpy scipy pybind11

**Building**

After installing the dependencies, you can build george by running:

.. code-block:: bash

    python -m pip install -e .

in the root directory of the source tree.

Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    python -m pytest -v tests

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/george/issues>`_.

Examples
--------

Take a look at :ref:`first` to get started and then check out the other
tutorials for some more advanced usage examples.
