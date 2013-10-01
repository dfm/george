#ifndef _GEORGE_H_
#define _GEORGE_H_

#include "cholmod.h"

typedef struct george_gp_struct {

    // The kernel function.
    double *pars;
    double (*kernel) (double, double, double*);

    // Bookkeeping flags.
    int computed, info;

    // The input data used to compute the factorization.
    int ndata;
    double *x;

    // The results of the factorization.
    double logdet;
    cholmod_common c;
    cholmod_factor *L;

} george_gp;

george_gp *george_allocate_gp
(
    double *pars,
    double (*kernel) (double, double, double*)
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

double george_kernel (
    double x1,
    double x2,
    double *pars
);

#endif
// /_GEORGE_H_
