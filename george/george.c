#include "math.h"
#include "george.h"

#define TWOLNPI 1.8378770664093453

//
// An extremely brittle function that take the internal representation of a
// CHOLMOD factorization and tries to compute the log-determinant.
//
// Based on code from CVXOPT and scikits-sparse.
//
double george_logdet (cholmod_factor *L, cholmod_common *c)
{
    int is_ll = L->is_ll, n = L->n, i, k;
    double v, logdet = 0.0, *x = L->x;

    if (L->is_super) {
        int nsuper = L->nsuper, *super = L->super, *pi = L->pi, *px = L->px,
            ncols, nrows, inc;
        for (i = 0; i < nsuper; i++){
            ncols = super[i+1] - super[i];
            nrows = pi[i+1] - pi[i];
            inc = nrows + 1;
            if (is_ll)
                for (k = 0; k < ncols * nrows; k += inc) {
                    v = x[px[i]+k];
                    logdet += log(v * v);
                }
            else
                for (k = 0; k < ncols * nrows; k += inc)
                    logdet += log(x[px[i]+k]);
        }
    } else {
        int *p = L->p;
        if (is_ll)
            for (i = 0; i < n; ++i) {
                v = x[p[i]];
                logdet += log(v * v);
            }
        else
            for (i = 0; i < n; ++i)
                logdet += log(x[p[i]]);
    }

    return logdet;
}

//
// The Gaussian process methods.
//
george_gp *
george_allocate_gp (double *pars, double (*kernel) (double, double, double*))
{
    george_gp *gp = malloc (sizeof (george_gp));

    gp->pars = pars;
    gp->kernel = kernel;
    gp->computed = 0;
    gp->info = 0;

    // Start up CHOLMOD.
    gp->c = malloc(sizeof(cholmod_common));
    cholmod_start (gp->c);
    gp->L = cholmod_allocate_factor (1, gp->c);

    return gp;
}

void george_free_gp (george_gp *gp)
{
    if (gp->computed) {
        free (gp->x);
        cholmod_free_factor (&(gp->L), gp->c);
    }
    cholmod_finish (gp->c);
    free (gp->c);
    free (gp);
}

int george_compute (int n, double *x, double *yerr, george_gp *gp)
{
    cholmod_common *c = gp->c;
    int i, j, k = 0, maxnnz = (n * n + n) / 2,
        *rows = malloc (maxnnz * sizeof(int)),
        *cols = malloc (maxnnz * sizeof(int));
    double value, *values = malloc (maxnnz * sizeof(double));

    // Compute the covariance matrix in triplet form.
    for (i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            value = (*(gp->kernel)) (x[i], x[j], gp->pars);
            if (i == j) value += yerr[i] * yerr[i];
            if (value > 0) {
                values[k] = value;
                rows[k] = i;
                cols[k] = j;
                ++k;
            }
        }
    }
    cholmod_triplet *triplet = cholmod_allocate_triplet (n, n, k, 1,
                                                         CHOLMOD_REAL, c);
    triplet->i = rows;
    triplet->j = cols;
    triplet->x = values;
    triplet->nnz = k;

    // Convert this to a sparse representation and compute the factorization.
    cholmod_sparse *A = cholmod_triplet_to_sparse (triplet, k, c);
    if (gp->computed) cholmod_free_factor (&(gp->L), c);
    gp->L = cholmod_analyze (A, c);
    gp->info = cholmod_factorize (A, gp->L, c);

    // Clean up.
    cholmod_free_sparse (&A, c);
    free(rows);
    free(cols);
    free(values);

    // Check the flag to make sure that the factorization occurred properly.
    if (!gp->info) return gp->info;

    // Pre-compute the log-determinant.
    gp->logdet = george_logdet (gp->L, c);

    // Save the data.
    gp->ndata = n;
    if (gp->computed) free(gp->x);
    gp->x = malloc(n * sizeof(double));
    for (i = 0; i < n; ++i) gp->x[i] = x[i];

    // Update the bookkeeping flags.
    gp->computed = 1;

    return gp->info;
}

double george_log_likelihood (double *y, george_gp *gp)
{
    int i, n = gp->ndata;
    cholmod_common *c = gp->c;

    // Make sure that things have been properly computed.
    if (!gp->computed) return -INFINITY;

    // Copy the column vector over.
    cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
    for (i = 0; i < n; ++i) ((double*)b->x)[i] = y[i];

    // Solve for alpha.
    cholmod_dense *alpha = cholmod_solve (CHOLMOD_A, gp->L, b, c);

    // Compute the log-likelihood.
    double lnlike = gp->logdet + gp->ndata * TWOLNPI, *ax = alpha->x;
    for (i = 0; i < n; ++i) lnlike += y[i] * ax[i];

    cholmod_free_dense (&b, c);
    cholmod_free_dense (&alpha, c);

    return -0.5 * lnlike;
}

//
// The built in kernel.
//
double george_kernel (double x1, double x2, double *pars)
{
    double d = x1 - x2, chi2 = d * d, r, omr, p2 = pars[2] * pars[2];
    if (chi2 >= p2) return 0.0;
    r = sqrt(chi2 / p2);
    omr = 1.0 - r;
    return pars[0] * pars[0] * exp(-0.5 * chi2 / (pars[1] * pars[1]))
           * omr * omr * (2*r + 1);
}
