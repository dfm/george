George
======

George is a fast and flexible Python library for Gaussian Process Regression.
A full introduction to the theory of Gaussian Processes is beyond the scope of
this documentation but the best resource is available for free online:
`Rasmussen & Williams (2006) <http://www.gaussianprocess.org/gpml/>`_.

George is being actively developed in `a public repository on GitHub
<https://github.com/dfm/george>`_ so if you have any trouble, `open an issue
<https://github.com/dfm/george/issues>`_ there.


User Guide
----------

.. toctree::
   :maxdepth: 2

   user/quickstart
   user/model
   user/hyper
   user/gp
   user/kernels
   user/solvers


License & Attribution
---------------------

Copyright 2014 Daniel Foreman-Mackey and contributors.

George is being developed by `Dan Foreman-Mackey <http://dan.iel.fm>`_ in a
`public GitHub repository <https://github.com/dfm/george>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite `the paper which is currently on the
ArXiv <http://arxiv.org/abs/1403.6015>`_:

.. code-block:: tex

    @article{hodlr,
        author = {{Ambikasaran}, S. and {Foreman-Mackey}, D. and
                  {Greengard}, L. and {Hogg}, D.~W. and {O'Neil}, M.},
         title = "{Fast Direct Methods for Gaussian Processes and the Analysis
                   of NASA Kepler Mission Data}",
          year = 2014,
         month = mar,
           url = http://arxiv.org/abs/1403.6015
    }
