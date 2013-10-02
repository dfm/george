George
======

**Blazingly fast Gaussian processes for regression.**

George is a C library (with Python bindings) that computes a [Gaussian
process](http://www.gaussianprocess.org/gpml/chapters/) regression model
taking full advantage of sparsity in the problem.
Right now, George is specifically optimized for the problem of fitting stellar
light curves produced by the Kepler satellite so it isn't yet generally
applicable.

Usage
-----

George is built on top of
[SuiteSparse](http://www.cise.ufl.edu/research/sparse/SuiteSparse/) a
state-of-the-art C library for sparse linear algebra.
Before installing George, make sure that you have a recent working version of
SuiteSparse.
In particular, George depends on a recent version of
[CHOLMOD](http://www.cise.ufl.edu/research/sparse/cholmod/) (a part of
SuiteSparse that computes Cholesky Decompositions) and it has only been tested
with version 2.1.2 (although it *might* work with earlier versions).

The build process uses [CMake](http://www.cmake.org/) so you'll need to
install that too.

After you have installed these dependencies, you should be able to just run:

```
cd /path/to/george/
mkdir build
cd build
cmake ..
make
make test
[sudo] make install
```

This will build and install both static (called `libgeorge`) and shared
(called `libgeorge_shared`) versions of the library.

License
-------

Copyright 2013 Daniel Foreman-Mackey

This is open source software licensed under the MIT license (see LICENSE).
