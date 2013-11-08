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

If you're on a Mac, you should probably be using [Homebrew](http://brew.sh/)
and if you are, you can install these dependencies by running:

```
brew install cmake
brew tap homebrew/science
brew install suite-sparse
```

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

If you ever want to uninstall George, just run:

```
cd /path/to/george/build
xargs rm < install_manifest.txt
```


API
---

The public interface to George is exposed through the `george_gp` type:

```
typedef struct george_gp_struct {

    int npars;              // The number of hyperparameters required by the
                            // kernel function.
    double *pars;           // The vector of hyperparameters.
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

    int ndata;              // The number of datapoints used to compute the
                            // covariance function and factorization. The
                            // kernel matrix K has dimension ndata x ndata.
    double *x, *yerr;       // The independent coordinates of the dataset used
                            // to compute the covariance matrix and the
                            // 1-sigma uncertainties in the data.

    int computed;           // A flag indicating the status of the
                            // factorization. When TRUE, the factorization
                            // has been computed and the L object exists.
    int info;               // A flag indicating the success or failure of
                            // the factorization.
    double logdet;          // The precomuted log-determinant of the
                            // covariance matrix.

    cholmod_common *c;      // The CHOLMOD workspace owned by this instance.
    cholmod_factor *L;      // The factorized representation of the matrix
                            // in CHOLMOD form.

} george_gp;
```

The following methods for working with this object are also provided.

**george_allocate_gp** — To create a new `george_gp` object, call:

```
george_gp *george_allocate_gp
(
    int npars,              // The number of hyperparameters.
    double *pars,           // The initial setting of the hyperparameters.
    void *meta,             // The metadata object for the kernel.
    double (*kernel)        // A function pointer to the kernel function.
        (double, double, double*, void*, int, double*, int*)
);
```

**george_free_gp** — To free the memory allocated by `george_allocate_gp`,
call:

```
void george_free_gp (
    george_gp *gp
);
```

**george_compute** — Given a dataset with errorbars, precompute the
factorization of the covariance matrix using:

```
int george_compute (
    int n,                  // The number of data points.
    double *x,              // The coordinates (timestamps) of the data points.
    double *yerr,           // The error bars on the datapoints.
    george_gp *gp           // The Gaussian process workspace.
);
```

**george_log_likelihood** — Compute the log-likelihood of a set of
observations `y` (after calling `george_compute`) using:

```
double george_log_likelihood (
    double *y,              // The observations (length: n).
    george_gp *gp           // The Gaussian process workspace.
);
```

**george_grad_log_likelihood** — Compute the log-likelihood and
gradient of the log-likelihood for a set of observations `y` (after
calling `george_compute`) using:

```
double george_grad_log_likelihood (
    double *y,              // The observations (length: n).
    double *grad_out,       // An output array giving the gradient of
                            // the log-likelihood for each parameter
                            // (length: npars).
    george_gp *gp           // The Gaussian process workspace.
);
```
**george_predict** — Evaluate the predictive distribution for a set of
independent coordinates `xout`:

```
int george_predict (
    double *y,              // The observations (length: n).
    int nout,               // The number of output points.
    double *xout,           // The input array of independent coordinates
                            // (length: nout).
    double *mean,           // The output predictive mean (length: nout).
    int compute_cov,        // Compute the output covariance matrix?
                            //   0: don't compute the covariance,
                            //   1: compute the covariance and store it
                            //      in cov.
    double *cov,            // The output covariance matrix
                            // (length: nout*nout).
    george_gp *gp           // The Gaussian process workspace.
);
```

**george_kernel** — An example sparse kernel function with 3 parameters.
It is the product of a squared exponential (with amplitude given by
`pars[0]*pars[0]` and variance `pars[1]*pars[1]`) and a polynomial with
support out to `pars[2]`.

```
double george_kernel (
    double x1, double x2,   // The coordinates.
    double *pars,           // The parameter vector (length: 3).
    void *meta,             // An arbitrary metadata object (unused).
    int compute_grad,       // Compute the gradient?
                            //   0: don't compute the gradient,
                            //   1: compute and store it in grad.
    double *grad,           // The output gradient vector (length: 3).
    int *flag               // Is there support at this distance?
                            //   0: the returned value is exactly zero,
                            //   1: a value was computed and it will be
                            //      >= 0.0.
);
```

License
-------

Copyright 2013 Daniel Foreman-Mackey

This is open source software licensed under the MIT license (see LICENSE).
