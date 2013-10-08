#include "math.h"
#include "george.h"

#define TWOLNPI 1.8378770664093453

void george_version (int *version)
{
    version[0] = GEORGE_VERSION_MAJOR;
    version[1] = GEORGE_VERSION_MINOR;
    version[2] = GEORGE_VERSION_PATCH;
}

void george_print_version ()
{
    printf ("George Gaussian process version: %d.%d.%d\n",
            GEORGE_VERSION_MAJOR,
            GEORGE_VERSION_MINOR,
            GEORGE_VERSION_PATCH);
}

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
george_allocate_gp (int npars, double *pars, void *meta,
                    double (*kernel) (double, double, double*, void*, int,
                                      double*, int*))
{
    george_gp *gp = malloc (sizeof (george_gp));

    gp->npars = npars;
    gp->pars = pars;
    gp->meta = meta;
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
        free (gp->yerr);
        cholmod_free_factor (&(gp->L), gp->c);
    }
    cholmod_finish (gp->c);
    free (gp->c);
    free (gp);
}

int george_compute (int n, double *x, double *yerr, george_gp *gp)
{
    cholmod_common *c = gp->c;
    int i, j, k = 0, maxnnz = (n * n + n) / 2, flag,
        *rows = malloc (maxnnz * sizeof(int)),
        *cols = malloc (maxnnz * sizeof(int));
    double value, *values = malloc (maxnnz * sizeof(double));

    // Compute the covariance matrix in triplet form.
    for (i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            value = (*(gp->kernel)) (x[i], x[j], gp->pars, gp->meta, 0, NULL,
                                     &flag);
            if (i == j) value += yerr[i] * yerr[i];
            if (flag && value > 0) {
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
    if (gp->computed) {
        free(gp->x);
        free(gp->yerr);
    }
    gp->x = malloc(n * sizeof(double));
    gp->yerr = malloc(n * sizeof(double));
    for (i = 0; i < n; ++i) {
        gp->x[i] = x[i];
        gp->yerr[i] = yerr[i];
    }

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

int george_grad_log_likelihood (double *y, double *grad_out, george_gp *gp)
{
    int i, j, k, flag, n = gp->ndata, npars = gp->npars;
    double value,
           *grad = malloc (npars * sizeof(double)),
           *x = gp->x;
    cholmod_common *c = gp->c;

    // Make sure that things have been properly computed.
    if (!gp->computed) return -1;

    // Copy the column vector over.
    cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
    for (i = 0; i < n; ++i) ((double*)b->x)[i] = y[i];

    // Solve for alpha.
    cholmod_dense *alpha = cholmod_solve (CHOLMOD_A, gp->L, b, c);
    double *alpha_data = (double*)alpha->x;
    cholmod_free_dense (&b, c);

    // Compute alpha.alpha^T.
    cholmod_dense *aat = cholmod_allocate_dense (n, n, n, CHOLMOD_REAL, c);
    double *aat_data = (double*)aat->x;
    for (i = 0; i < n; ++i) {
        aat_data[i*n+i] = alpha_data[i] * alpha_data[i];
        for (j = i+1; j < n; ++j) {
            value = alpha_data[i] * alpha_data[j];
            aat_data[i*n+j] = value;
            aat_data[j*n+i] = value;
        }
    }
    cholmod_free_dense (&alpha, c);

    // Allocate memory for the kernel matrix gradients.
    cholmod_dense **dkdt = malloc(npars*sizeof(cholmod_dense*));
    double **dkdt_data = malloc(npars*sizeof(double*));

    for (k = 0; k < npars; ++k) {
        dkdt[k] = cholmod_allocate_dense (n, n, n, CHOLMOD_REAL, c);
        dkdt_data[k] = (double*)dkdt[k]->x;
    }

    // Loop over the data points and compute the kernel matrix gradients.
    for (i = 0; i < n; ++i) {
        // Compute the diagonal terms.
        (*(gp->kernel)) (x[i], x[i], gp->pars, gp->meta, 1, grad, &flag);
        if (flag)
            for (k = 0; k < npars; ++k) dkdt_data[k][i*n+i] = grad[k];
        else
            for (k = 0; k < npars; ++k) dkdt_data[k][i*n+i] = 0.0;

        // And the off-diagonal terms.
        for (j = i+1; j < n; ++j) {
            (*(gp->kernel)) (x[i], x[j], gp->pars, gp->meta, 1, grad, &flag);
            if (flag)
                for (k = 0; k < npars; ++k) {
                    dkdt_data[k][i*n+j] = grad[k];
                    dkdt_data[k][j*n+i] = grad[k];
                }
            else
                for (k = 0; k < npars; ++k) {
                    dkdt_data[k][i*n+j] = 0.0;
                    dkdt_data[k][j*n+i] = 0.0;
                }
        }
    }
    free (grad);

    // Loop over each hyperparameter and solve for the gradient.
    int ind;
    cholmod_dense *kdkdt;
    double *kdkdt_data;
    for (k = 0; k < npars; ++k) {
        // Solve the system.
        kdkdt = cholmod_solve (CHOLMOD_A, gp->L, dkdt[k], c);
        kdkdt_data = (double*)kdkdt->x;

        // Take the trace.
        grad_out[k] = 0.0;
        for (i = 0; i < n; ++i) {
            grad_out[k] -= kdkdt_data[i*n+i];
            for (j = 0; j < n; ++j) {
                ind = i*n+j;
                grad_out[k] += aat_data[ind] * dkdt_data[k][ind];
            }
        }
        grad_out[k] *= 0.5;

        cholmod_free_dense (&dkdt[k], c);
        cholmod_free_dense (&kdkdt, c);
    }

    cholmod_free_dense (&aat, c);

    return 0;
}

int george_predict (double *y, int nout, double *xout, double *mean,
                    int compute_cov, double *cov, george_gp *gp)
{
    cholmod_common *c = gp->c;
    int i, j, k, n = gp->ndata, flag;
    double value;

    if (!gp->computed) return -1;

    // Copy the column vector over.
    cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
    for (i = 0; i < n; ++i) ((double*)b->x)[i] = y[i];

    // Solve for alpha.
    cholmod_dense *alpha = cholmod_solve (CHOLMOD_A, gp->L, b, c);
    double *alpha_data = (double*)alpha->x;
    cholmod_free_dense (&b, c);

    // Initialize the mean vector as zeros.
    for (i = 0; i < nout; ++i) mean[i] = 0.0;

    // Loop over the data points and compute the Kxs matrix.
    cholmod_dense *kxs = cholmod_allocate_dense (n, nout, nout, CHOLMOD_REAL, c);
    double *kxs_data = (double*)kxs->x,
           *x = gp->x;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < nout; ++j) {
            value = (*(gp->kernel)) (x[i], xout[j], gp->pars, gp->meta, 0,
                                     NULL, &flag);
            if (!flag) value = 0.0;
            kxs_data[i*nout + j] = value;
            mean[j] += value * alpha_data[i];
        }
    }
    cholmod_free_dense (&alpha, c);

    if (!compute_cov) return 0;

    // Initialize the output covariance.
    for (i = 0; i < nout; ++i) {
        value = (*(gp->kernel)) (xout[i], xout[i], gp->pars, gp->meta, 0,
                                 NULL, &flag);
        if (flag) cov[i*nout+i] = value;
        else cov[i*nout+i] = 0.0;
        for (j = i + 1; j < nout; ++j) {
            value = (*(gp->kernel)) (xout[i], xout[j], gp->pars, gp->meta, 0,
                                     NULL, &flag);
            if (!flag) value = 0.0;
            cov[i*nout+j] = value;
            cov[j*nout+i] = value;
        }
    }

    // Update the covariance matrix.
    cholmod_dense *tmp = cholmod_solve (CHOLMOD_A, gp->L, kxs, c);
    double *tmp_data = (double*)tmp->x;
    for (i = 0; i < nout; ++i) {
        for (j = 0; j < nout; ++j) {
            for (k = 0; k < n; ++k) {
                cov[i*nout+j] -= kxs_data[k*nout+i] * tmp_data[j*nout+k];
            }
        }
    }
    cholmod_free_dense (&tmp, c);

    return 0;
}

//
// The built in kernel.
//
double george_kernel (double x1, double x2, double *pars, void *meta,
                      int compute_grad, double *grad, int *flag)
{
    double d = x1 - x2, chi2 = d * d, r, omr, k0, k,
           *p = pars,
           p2 = p[2] * p[2],
           p1 = p[1] * p[1];

    // If the distance is greater than the support, bail.
    *flag = 0;
    if (chi2 >= p2) return 0.0;

    // Compute the kernel value.
    *flag = 1;
    r = sqrt(chi2 / p2);
    omr = 1.0 - r;
    k0 = p[0] * p[0] * exp(-0.5 * chi2 / p1);
    k = k0 * omr * omr * (2*r + 1);

    // Compute the gradient.
    if (compute_grad) {
        grad[0] = 2 * k / p[0];
        grad[1] = k * chi2 / (p[1] * p1);
        grad[2] = 6 * k0 * omr * r * r / p[2];
    }

    return k;
}
