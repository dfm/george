#ifndef _GEORGE_H_
#define _GEORGE_H_

#include "cholmod.h"

#define GEORGE_VERSION_MAJOR    0
#define GEORGE_VERSION_MINOR    1
#define GEORGE_VERSION_RELEASE  0

void george_version (int *version);
void george_print_version ();

typedef struct george_gp_struct {

    // The kernel function.
    int npars;
    double *pars;
    void *meta;
    double (*kernel) (double, double, double*, void*, int, double*, int*);

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

george_gp *george_allocate_gp
(
    int npars,
    double *pars,
    void *meta,
    double (*kernel) (double, double, double*, void*, int, double*, int*)
);

void george_free_gp (
    george_gp *gp
);

int george_compute (
    int n,
    double *x,
    double *yerr,
    george_gp *gp
);

double george_log_likelihood (
    double *y,
    george_gp *gp
);

int george_grad_log_likelihood (
    double *y,
    double *grad_out,
    george_gp *gp
);

double george_kernel (
    double x1,
    double x2,
    double *pars,
    void *meta,
    int compute_grad,
    double *grad,
    int *flag
);

#endif
// /_GEORGE_H_
