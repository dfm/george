#ifndef _GEORGE_SOLVER_H_
#define _GEORGE_SOLVER_H_

#include <cmath>
#include <Eigen/Dense>
#include <HODLR_Tree.hpp>
#include <HODLR_Matrix.hpp>

#include "constants.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace george {

template <typename K>
class HODLRSolverMatrix : public HODLR_Matrix {

public:

    HODLRSolverMatrix (K* kernel) : kernel_(kernel) {};
    void set_values (MatrixXd v) { t_ = v; };
    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return kernel_->evaluate (t_.row(i), t_.row(j));
    };

private:

    K* kernel_;
    MatrixXd t_;

};

template <typename K>
class HODLRSolver {

public:

    //
    // Allocation and deallocation.
    //
    HODLRSolver (K* kernel, unsigned nLeaf = 10, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), kernel_(kernel)
    {
        matrix_ = new HODLRSolverMatrix<K> (kernel_);
        solver_ = NULL;
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~HODLRSolver () {
        if (solver_ != NULL) delete solver_;
        delete matrix_;
    };

    //
    // Flag accessors.
    //
    int get_status () const { return status_; };
    int get_computed () const { return computed_; };

    //
    // Get the dimensions.
    //
    int get_dimension () const {
        if (computed_) return x_.rows();
        return -1;
    };

    //
    // Pre-compute and factorize the kernel matrix.
    //
    int compute (const MatrixXd x, const VectorXd& yerr, int seed) {
        // Check the dimensions.
        int n = x.rows();
        if (yerr.rows() != n) {
            status_ = DIMENSION_MISMATCH;
            return status_;
        }

        // It's not computed until it's computed...
        computed_ = 0;

        // Compute the diagonal elements.
        VectorXd diag(n);
        for (int i = 0; i < n; ++i)
            diag[i] = yerr[i]*yerr[i] +
                      kernel_->evaluate(x.row(i), x.row(i));

        // Set the time points for the kernel.
        matrix_->set_values (x);

        // Set up the solver object.
        if (solver_ != NULL) delete solver_;
        srand(seed);
        solver_ = new HODLR_Tree<HODLRSolverMatrix<K> > (matrix_, n, nleaf_);
        solver_->assemble_Matrix(diag, tol_);

        // Factorize the matrix.
        solver_->compute_Factor();

        // Extract the log-determinant.
        solver_->compute_Determinant(logdet_);
        logdet_ += n * log(2 * M_PI);

        // Save the data for later use.
        x_ = x;

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
        // Make sure that things have been properly computed.
        if (!computed_ || status_ != SOLVER_OK) {
            status_ = USAGE_ERROR;
            return -INFINITY;
        }

        // Check the dimensions.
        if (y.rows() != x_.rows()) {
            status_ = DIMENSION_MISMATCH;
            return -INFINITY;
        }

        // Compute the log-likelihood.
        return -0.5 * (logdet_ + y.dot(compute_alpha(y)));
    };

    //
    // Compute the ``alpha`` vector for a given RHS.
    //
    VectorXd compute_alpha (MatrixXd y) {
        MatrixXd alpha(y.rows(), 1);
        solver_->solve(y, alpha);
        return alpha;
    };

    //
    // Get the mean conditional prediction.
    //
    void predict (const VectorXd& y, const MatrixXd& t, VectorXd& mu, MatrixXd& cov) {
        int n = y.rows(), nt = t.rows(), ndim = t.cols();
        if (n != x_.rows() || ndim != x_.cols()) {
            status_ = DIMENSION_MISMATCH;
            return;
        }

        // Compute the cross kernel matrix.
        MatrixXd k (nt, n);
        for (int i = 0; i < nt; ++i)
            for (int j = 0; j < n; ++j)
                k(i, j) = kernel_->evaluate (t.row(i), x_.row(j));

        // Compute the mean prediction.
        mu = k * compute_alpha(y);

        // Compute the predictive covariance.
        cov = MatrixXd (nt, nt);
        for (int i = 0; i < nt; ++i)
            for (int j = 0; j < nt; ++j)
                cov(i, j) = kernel_->evaluate (t.row(i), t.row(j));
        MatrixXd v(nt, nt),
                 kt = k.transpose();
        solver_->solve(kt, v);
        cov -= k * v;
    };

private:

    double logdet_, tol_;
    unsigned nleaf_;
    int status_, computed_;
    K* kernel_;
    HODLRSolverMatrix<K>* matrix_;
    HODLR_Tree<HODLRSolverMatrix<K> >* solver_;
    MatrixXd x_;

};

};

#endif
