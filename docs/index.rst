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
    :target: https://github.com/dfm/george/blob/main/LICENSE
.. image:: https://github.com/dfm/george/workflows/Tests/badge.svg?style=flat
    :target: https://github.com/dfm/george/actions?query=workflow%3ATests
.. image:: https://coveralls.io/repos/github/dfm/george/badge.svg?branch=main&style=flat
    :target: https://coveralls.io/github/dfm/george?branch=main


.. toctree::
   :maxdepth: 2

   user/index
   tutorials/index


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2012-2023 Daniel Foreman-Mackey and contributors.

George is being developed by `Dan Foreman-Mackey <https://dfm.io>`_ in a
`public GitHub repository <https://github.com/dfm/george>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite `the paper which is in
IEEE Transactions on Pattern Analysis and Machine Intelligence
<https://ui.adsabs.harvard.edu/abs/2015ITPAM..38..252A/abstract>`_:

.. code-block:: tex

   @ARTICLE{2015ITPAM..38..252A,
           author = {{Ambikasaran}, Sivaram and {Foreman-Mackey}, Daniel and {Greengard}, Leslie and {Hogg}, David W. and {O'Neil}, Michael},
            title = "{Fast Direct Methods for Gaussian Processes}",
          journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
         keywords = {Mathematics - Numerical Analysis, Astrophysics - Instrumentation and Methods for Astrophysics, Mathematics - Statistics Theory, Mathematics - Numerical Analysis, Astrophysics - Instrumentation and Methods for Astrophysics, Mathematics - Statistics Theory},
             year = 2015,
            month = jun,
           volume = {38},
            pages = {252},
              doi = {10.1109/TPAMI.2015.2448083},
    archivePrefix = {arXiv},
           eprint = {1403.6015},
     primaryClass = {math.NA},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2015ITPAM..38..252A},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }


Changelog
---------

.. include:: ../HISTORY.rst
