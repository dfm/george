#include "george.h"
#include <math.h>
#include <lbfgs.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

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
    if (gp == NULL) return NULL;

    gp->npars = npars;
    gp->pars = pars;
    gp->meta = meta;
    gp->kernel = kernel;
    gp->computed = 0;
    gp->info = CHOLMOD_OK;
    gp->c = malloc(sizeof(cholmod_common));
    if (gp->c == NULL) {
        free(gp);
        return NULL;
    }

    // Start up CHOLMOD.
    cholmod_start (gp->c);
    if (gp->c->status != CHOLMOD_OK || !cholmod_check_common(gp->c)) {
        free(gp->c);
        free(gp);
        return NULL;
    }

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
    int i, j, k = 0, maxnnz = (n * n + n) / 2, flag;
    double value;

    // Initialize the `computed` flag and clean up any existing data.
    if (gp->computed) {
        cholmod_free_factor (&(gp->L), c);
        free(gp->x);
        free(gp->yerr);
    }
    gp->computed = 0;

    // Allocate some memory for a sparse triplet matrix.
    cholmod_triplet *triplet = cholmod_allocate_triplet (n, n, maxnnz, 1,
                                                         CHOLMOD_REAL, c);
    if (triplet == NULL || !cholmod_check_triplet(triplet, c)) {
        if (triplet != NULL) cholmod_free_triplet(&triplet, c);
        gp->info = -1;
        return gp->info;
    }

    // Compute the covariance matrix in triplet form.
    for (i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            value = (*(gp->kernel)) (x[i], x[j], gp->pars, gp->meta, 0, NULL,
                                     &flag);
            if (i == j) value += yerr[i] * yerr[i];
            if (flag && value > 0) {
                ((double*)triplet->x)[k] = value;
                ((int*)triplet->i)[k] = i;
                ((int*)triplet->j)[k] = j;
                ++k;
            }
        }
    }

    // Save the final number of non-zero values.
    triplet->nnz = k;

    // Convert this to a sparse representation.
    cholmod_sparse *A = cholmod_triplet_to_sparse (triplet, k, c);
    cholmod_free_triplet(&triplet, c);
    if (A == NULL || !cholmod_check_sparse(A, c)) {
        gp->info = -2;
        return gp->info;
    }

    // Analyze the covariance matrix to find the best factorization pattern.
    gp->L = cholmod_analyze (A, c);
    if (gp->L == NULL) {
        cholmod_free_sparse (&A, c);
        gp->info = -3;
        return gp->info;
    }

    // Check the success of this analysis.
    gp->info = c->status;
    if (gp->info != CHOLMOD_OK || !cholmod_check_factor(gp->L, c)) {
        cholmod_free_sparse (&A, c);
        gp->info = -4;
        return gp->info;
    }

    // Factorize the matrix.
    cholmod_factorize (A, gp->L, c);
    gp->info = c->status;
    cholmod_free_sparse (&A, c);
    if (gp->info != CHOLMOD_OK || !cholmod_check_factor(gp->L, c)) {
        gp->info = -5;
        return gp->info;
    }

    // Pre-compute the log-determinant.
    gp->logdet = george_logdet (gp->L, c) + n * log(2 * M_PI);

    // Save the data for later use.
    gp->ndata = n;
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

cholmod_dense *_george_get_alpha (double *y, george_gp *gp)
{
    int i, n = gp->ndata;
    cholmod_common *c = gp->c;

    // Make sure that things have been properly computed.
    if (!gp->computed || gp->info != CHOLMOD_OK) return NULL;

    // Copy the column vector over to a dense matrix.
    cholmod_dense *b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
    if (b == NULL) {
        gp->info = -1;
        return NULL;
    }
    if (!cholmod_check_dense(b, c)) {
        cholmod_free_dense(&b, c);
        gp->info = -2;
        return NULL;
    }
    for (i = 0; i < n; ++i) ((double*)b->x)[i] = y[i];

    // Solve for alpha.
    cholmod_dense *alpha = cholmod_solve (CHOLMOD_A, gp->L, b, c);
    cholmod_free_dense (&b, c);

    // Check the success of the solve.
    gp->info = c->status;
    if (gp->info != CHOLMOD_OK) {
        cholmod_free_dense (&alpha, c);
        return NULL;
    }

    return alpha;
}

double _george_compute_log_likelihood (double *y, cholmod_dense *alpha,
                                       george_gp *gp)
{
    int i, n = gp->ndata;
    double lnlike = gp->logdet, *ax = alpha->x;
    for (i = 0; i < n; ++i) lnlike += y[i] * ax[i];
    return -0.5*lnlike;
}

double george_log_likelihood (double *y, george_gp *gp)
{
    double lnlike;
    cholmod_dense *alpha = _george_get_alpha (y, gp);
    if (alpha == NULL) return -INFINITY;
    lnlike = _george_compute_log_likelihood (y, alpha, gp);
    cholmod_free_dense (&alpha, gp->c);
    return lnlike;
}

double george_grad_log_likelihood (double *y, double *grad_out, george_gp *gp)
{
    int i, j, k, flag, n = gp->ndata, npars = gp->npars;
    cholmod_common *c = gp->c;
    cholmod_dense *alpha, *aat, **dkdt;
    double *alpha_data, *aat_data, **dkdt_data,
           lnlike, value, *grad, *x = gp->x;

    // Solve for alpha.
    alpha = _george_get_alpha (y, gp);
    if (alpha == NULL) return -INFINITY;
    alpha_data = (double*)alpha->x;

    // Compute the log-likelihood.
    lnlike = _george_compute_log_likelihood (y, alpha, gp);

    // Compute alpha.alpha^T.
    aat = cholmod_allocate_dense (n, n, n, CHOLMOD_REAL, c);
    if (aat == NULL || !cholmod_check_dense (alpha, c)) {
        if (aat != NULL) cholmod_free_dense(&aat, c);
        cholmod_free_dense (&alpha, c);
        gp->info = -2;
        return -INFINITY;
    }
    aat_data = (double*)aat->x;
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
    dkdt = malloc(npars*sizeof(cholmod_dense*));
    dkdt_data = malloc(npars*sizeof(double*));
    for (k = 0; k < npars; ++k) {
        dkdt[k] = cholmod_allocate_dense (n, n, n, CHOLMOD_REAL, c);
        dkdt_data[k] = (double*)dkdt[k]->x;
    }

    // Loop over the data points and compute the kernel matrix gradients.
    grad = malloc (npars * sizeof(double));
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

    return lnlike;
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
    cholmod_dense *kxs = cholmod_allocate_dense (n, nout, n, CHOLMOD_REAL, c);
    double *kxs_data = (double*)kxs->x,
           *x = gp->x;

    // Note: column major order.
    for (j = 0; j < nout; ++j) {
        for (i = 0; i < n; ++i) {
            value = (*(gp->kernel)) (x[i], xout[j], gp->pars, gp->meta, 0,
                                     NULL, &flag);
            if (!flag) value = 0.0;
            kxs_data[i + j*n] = value;
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
    cholmod_dense *v = cholmod_solve (CHOLMOD_A, gp->L, kxs, c);
    double *v_data = (double*)v->x;
    for (i = 0; i < nout; ++i) {
        for (k = 0; k < n; ++k)
            cov[i*nout+i] -= kxs_data[k+i*n] * v_data[k+i*n];
        for (j = i+1; j < nout; ++j) {
            for (k = 0; k < n; ++k) {
                value = kxs_data[k+i*n] * v_data[k+j*n];
                cov[i*nout+j] -= value;
                cov[j*nout+i] -= value;
            }
        }
    }
    cholmod_free_dense (&v, c);

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

//
// Optimization of hyperparameters.
//

typedef struct _george_op_wrapper_struct {
    int n, verbose;
    double *x, *yerr, *y;
    george_gp *gp;
} _george_op_wrapper;

static
lbfgsfloatval_t _george_op_evaluate(void *instance, const lbfgsfloatval_t *w,
                                    lbfgsfloatval_t *grad, const int n,
                                    const lbfgsfloatval_t step)
{
    int i, info;
    double nlp;
    _george_op_wrapper *wrapper = (_george_op_wrapper*)instance;
    wrapper->gp->pars = (double*)w;
    info = george_compute(wrapper->n, wrapper->x, wrapper->yerr, wrapper->gp);
    nlp = -george_grad_log_likelihood(wrapper->y, (double*)grad, wrapper->gp);
    for (i = 0; i < n; ++i) grad[i] = -grad[i];
    if (info != 0) return INFINITY;
    return nlp;
}

static
int _george_op_progress(void *instance, const lbfgsfloatval_t *x,
             const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
             const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
             const lbfgsfloatval_t step, int n, int k, int ls)
{
    int i;
    if (((_george_op_wrapper*)instance)->verbose) {
        printf("Iteration %d: ", k);
        printf("fx = %f\n", fx);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("  ");
        for (i = 0; i < n; ++i) printf("%e ", x[i]);
        printf("\n\n");
    }
    return 0;
}

int george_optimize (int n, double *x, double *yerr, double *y, int maxiter,
                     int verbose, george_gp *gp)
{
    int i;

    _george_op_wrapper *wrapper = malloc(sizeof(_george_op_wrapper));
    wrapper->n = n;
    wrapper->verbose = verbose;
    wrapper->x = x;
    wrapper->yerr = yerr;
    wrapper->y = y;
    wrapper->gp = gp;

    double *initial_pars = gp->pars;

    lbfgsfloatval_t fx;
    lbfgsfloatval_t *xval = lbfgs_malloc(gp->npars);
    lbfgs_parameter_t param;
    for (i = 0; i < gp->npars; ++i) xval[i] = initial_pars[i];

    lbfgs_parameter_init(&param);
    param.max_iterations = maxiter;
    int r = lbfgs(gp->npars, xval, &fx, _george_op_evaluate,
                  _george_op_progress, wrapper, &param);
    if (verbose) {
        printf("L-BFGS optimization terminated with status code = %d\n", r);
        printf("  fx = %f\n", fx);
    }

    for (i = 0; i < gp->npars; ++i) initial_pars[i] = xval[i];
    gp->pars = initial_pars;

    lbfgs_free(xval);
    free(wrapper);
    return r;
}
