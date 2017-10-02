George
======

George is a fast and flexible Python library for Gaussian Process (GP)
Regression. A full introduction to the theory of Gaussian Processes is beyond
the scope of this documentation but the best resource is available for free
online: `Rasmussen & Williams (2006) <http://www.gaussianprocess.org/gpml/>`_.

Unlike some other GP implementations, george is focused on efficiently
evaluating the marginalized likelihood of a dataset under a GP prior, even as
this dataset gets Big™. As you'll see in these pages of documentation, the
module exposes quite a few other features but it is designed to be used
alongside your favorite `non-linear optimization
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
or `posterior inference <http://dfm.io/emcee>`_ library for the best results.

George is being actively developed in `a public repository on GitHub
<https://github.com/dfm/george>`_ so if you have any trouble, `open an issue
<https://github.com/dfm/george/issues>`_ there.

.. image:: https://img.shields.io/badge/GitHub-dfm%2Fgeorge-blue.svg?style=flat
    :target: https://github.com/dfm/george
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/dfm/george/blob/master/LICENSE
.. image:: http://img.shields.io/travis/dfm/george/master.svg?style=flat
    :target: https://travis-ci.org/dfm/george
.. image:: https://ci.appveyor.com/api/projects/status/xy4ts3v3sk5lo5ll?svg=true&style=flat
    :target: https://ci.appveyor.com/project/dfm/george
.. image:: https://coveralls.io/repos/github/dfm/george/badge.svg?branch=master&style=flat
    :target: https://coveralls.io/github/dfm/george?branch=master


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/quickstart
   user/kernels
   user/gp
   user/solvers
   user/modeling


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/first
   tutorials/model
   tutorials/hyper
   tutorials/scaling
   tutorials/new-kernel
   tutorials/mixture
   tutorials/bayesopt


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2012-2017 Daniel Foreman-Mackey and contributors.

George is being developed by `Dan Foreman-Mackey <http://dan.iel.fm>`_ in a
`public GitHub repository <https://github.com/dfm/george>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite `the paper which is currently on the
ArXiv <http://arxiv.org/abs/1403.6015>`_:

.. code-block:: tex

    @article{hodlr,
        author = {{Ambikasaran}, S. and {Foreman-Mackey}, D. and
                  {Greengard}, L. and {Hogg}, D.~W. and {O'Neil}, M.},
         title = "{Fast Direct Methods for Gaussian Processes}",
          year = 2014,
         month = mar,
           url = http://arxiv.org/abs/1403.6015
    }


Changelog
---------

.. include:: ../HISTORY.rst
