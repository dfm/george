#ifndef _GEORGE_SPARSE_SOLVER_H_
#define _GEORGE_SPARSE_SOLVER_H_

#include <cmath>
#include <cstdlib>
#include <cholmod.h>
#include <Eigen/Dense>

#include "constants.h"

using Eigen::VectorXd;

namespace george {

template <typename K>
class SparseSolver {

public:

    //
    // Allocation and deallocation.
    //
    SparseSolver (K* kernel) : kernel_(kernel) {
        status_ = SOLVER_OK;
        computed_ = 0;
        L_ = NULL;

        // Allocate and set up the CHOLMOD workspace.
        common_ = (cholmod_common*)malloc(sizeof(cholmod_common));
        cholmod_start (common_);
        if (common_->status != CHOLMOD_OK || !cholmod_check_common(common_)) {
            status_ = SETUP_FAILURE;
            free(common_);
            return;
        }
    };

    ~SparseSolver () {
        if (computed_ && L_ != NULL) cholmod_free_factor (&L_, common_);
        cholmod_finish (common_);
        free (common_);
    };

    //
    // Flag accessors.
    //
    int get_status () const { return status_; };
    int get_cholmod_status () const { return common_->status; };
    int get_computed () const { return computed_; };

    //
    // Compute the GP.
    //
    int compute (VectorXd x, VectorXd yerr) {
        int i, j, k = 0, n = x.rows(), maxnnz = (n * n + n) / 2, flag;
        double value;

        // Check the dimensions.
        if (yerr.rows() != n) {
            status_ = DIMENSION_MISMATCH;
            return status_;
        }

        // Initialize the `computed` flag and clean up any existing data.
        if (computed_) cholmod_free_factor (&L_, common_);
        computed_ = 0;

        // Allocate some memory for a sparse triplet matrix.
        cholmod_triplet* triplet = cholmod_allocate_triplet (n, n, maxnnz, 1,
                                                             CHOLMOD_REAL,
                                                             common_);
        if (triplet == NULL || !cholmod_check_triplet(triplet, common_)) {
            if (triplet != NULL) cholmod_free_triplet(&triplet, common_);
            status_ = CHOLMOD_ERROR;
            return status_;
        }

        // Compute the covariance matrix in triplet form.
        for (i = 0; i < n; ++i) {
            for (j = i; j < n; ++j) {
                value = kernel_->evaluate(x[i], x[j], 0, NULL, &flag);
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
        cholmod_sparse* A = cholmod_triplet_to_sparse (triplet, k, common_);
        cholmod_free_triplet(&triplet, common_);
        if (A == NULL || !cholmod_check_sparse(A, common_)) {
            status_ = CHOLMOD_ERROR;
            return status_;
        }

        // Check the success of this analysis.
        L_ = cholmod_analyze (A, common_);
        if (common_->status != CHOLMOD_OK || L_ == NULL ||
                !cholmod_check_factor(L_, common_)) {
            if (L_ != NULL) cholmod_free_sparse (&A, common_);
            status_ = CHOLMOD_ERROR;
            return status_;
        }

        // Factorize the matrix.
        cholmod_factorize (A, L_, common_);
        cholmod_free_sparse (&A, common_);
        if (common_->status != CHOLMOD_OK || !cholmod_check_factor(L_, common_)) {
            status_ = CHOLMOD_ERROR;
            return status_;
        }

        // Pre-compute the log-determinant.
        logdet_ = extract_logdet() + n * log(2 * M_PI);

        // Save the data for later use.
        x_ = x;
        yerr_ = yerr;

        // Update the bookkeeping flags.
        computed_ = 1;
        status_ = SOLVER_OK;
        return status_;
    };

    //
    // Compute the log-likelihood
    //
    double log_likelihood (VectorXd y)
    {
        // Pre-compute the ``alpha`` vector.
        cholmod_dense* alpha = compute_alpha (y);
        if (alpha == NULL) return -INFINITY;

        // Compute the log-likelihood using the private function.
        double lnlike = log_likelihood (y, alpha);
        cholmod_free_dense (&alpha, common_);
        return lnlike;
    };

private:

    double logdet_;
    int status_, computed_;
    cholmod_common* common_;
    cholmod_factor* L_;
    K* kernel_;
    VectorXd x_, yerr_;

    //
    // Extract the log determinant from the CHOLMOD factor.
    //
    double extract_logdet () const {
        int is_ll = L_->is_ll, n = L_->n, i, k;
        double v, logdet = 0.0, *x = (double*)(L_->x);

        if (L_->is_super) {
            int nsuper = L_->nsuper, *super = (int*)(L_->super),
                *pi = (int*)(L_->pi), *px = (int*)(L_->px), ncols, nrows, inc;
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
            int* p = (int*)(L_->p);
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
    };

    //
    // Compute the ``alpha`` vector for a given RHS.
    //
    cholmod_dense* compute_alpha (VectorXd y) {
        int i, n = y.rows();

        // Make sure that things have been properly computed.
        if (!computed_ || status_ != SOLVER_OK || L_ == NULL) return NULL;

        // Copy the column vector over to a dense matrix.
        cholmod_dense* b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, common_);
        if (b == NULL) {
            status_ = MEMORY_ERROR;
            return NULL;
        }
        if (!cholmod_check_dense(b, common_)) {
            cholmod_free_dense(&b, common_);
            status_ = CHOLMOD_ERROR;
            return NULL;
        }
        for (i = 0; i < n; ++i) ((double*)b->x)[i] = y[i];

        // Solve for alpha.
        cholmod_dense* alpha = cholmod_solve (CHOLMOD_A, L_, b, common_);
        cholmod_free_dense (&b, common_);

        // Check the success of the solve.
        if (common_->status != CHOLMOD_OK) {
            cholmod_free_dense (&alpha, common_);
            status_ = CHOLMOD_ERROR;
            return NULL;
        }

        return alpha;
    };

    //
    // Compute the log-likelihood for a known value of ``alpha``.
    //
    double log_likelihood (VectorXd y, cholmod_dense* alpha) {
        // Make sure that the system has been computed.
        if (!computed_) {
            status_ = USAGE_ERROR;
            return -INFINITY;
        }

        // Check the dimensions.
        int n = y.rows();
        if (n != x_.rows()) {
            status_ = DIMENSION_MISMATCH;
            return -INFINITY;
        }

        // Compute the likelihood.
        double lnlike = logdet_, *ax = (double*)(alpha->x);
        for (int i = 0; i < n; ++i) lnlike += y[i] * ax[i];
        return -0.5 * lnlike;
    };

};

};

#endif
