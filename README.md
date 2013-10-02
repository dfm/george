George
======

**Blazingly fast Gaussian processes for regression.**

George is a C library (with Python bindings) that computes a [Gaussian
process](http://www.gaussianprocess.org/gpml/chapters/) regression model
taking full advantage of sparsity in the problem.
Right now, George is specifically optimized for the problem of fitting stellar
light curves produced by the Kepler satellite so it isn't yet generally
applicable.


Installation
------------

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


API
---

The main public interface to George is the `george_gp` type:

```
typedef struct george_gp_struct {

    //
    // The kernel function.
    //
    int npars;              // The number of hyperparameters.
    double *pars;           // A vector of hyperparameters.
    void *meta;             // Any metadata that the kernel might need.
    double (*kernel) (      // A pointer to the kernel function.
        // -- inputs --
        double x1,          // The first coordinate.
        double x2,          // The second coordinate.
        double *pars,       // The hyperparameter vector.
        void *meta,         // The metadata object.
        int compute_grad,   // A flag indicating whether or not the gradient
                            // should be computed.
                            // FALSE: don't compute the gradient and *grad is
                            //        probably NULL.
                            // TRUE:  compute the gradient and store it in
                            //        *grad.
        // -- outputs --
        double *grad,       // The gradient vector or NULL.
        int *flag           // A flag indicating that the kernel was non-zero.
    );

    // Bookkeeping flags.
    int computed, info;

    // The input data used to compute the factorization.
    int ndata;
    double *x, *yerr;

    // The results of the factorization.
    double logdet;
    cholmod_common *c;
    cholmod_factor *L;

} george_gp;
```


License
-------

Copyright 2013 Daniel Foreman-Mackey

This is open source software licensed under the MIT license (see LICENSE).
