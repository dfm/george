#ifndef _GEORGE_HODLR_SOLVER_H_
#define _GEORGE_HODLR_SOLVER_H_

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
    void set_values (VectorXd v) { t_ = v; };
    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        int flag = 0;
        double v = kernel_->evaluate (t_[i], t_[j], 0, NULL, &flag);
        if (flag) return v;
        return 0.0;
    };

private:

    K* kernel_;
    VectorXd t_;

};

template <typename K>
class HODLRSolver {

public:

    //
    // Allocation and deallocation.
    //
    HODLRSolver (K* kernel, unsigned nLeaf = 50, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), kernel_(kernel)
    {
        matrix_ = new HODLRSolverMatrix<K> (kernel_);
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~HODLRSolver () {
        delete matrix_;
    };

    //
    // Flag accessors.
    //
    int get_status () const { return status_; };
    int get_computed () const { return computed_; };

    //
    // Pre-compute and factorize the kernel matrix.
    //
    int compute (VectorXd x, VectorXd yerr) {
        int flag;

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
            diag[i] = yerr[i]*yerr[i] + kernel_->evaluate(x[i], x[i], 0, NULL, &flag);

        // Set the time points for the kernel.
        matrix_->set_values (x);

        // Set up the solver object.
        solver_ = new HODLR_Tree<HODLRSolverMatrix<K> > (matrix_, n, nleaf_);
        solver_->assemble_Matrix(diag, tol_);

        // Factorize the matrix.
        solver_->compute_Factor();

        // Extract the log-determinant.
        solver_->compute_Determinant(logdet_);
        logdet_ += n * log(2 * M_PI);

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

private:

    double logdet_, tol_;
    unsigned nleaf_;
    int status_, computed_;
    K* kernel_;
    HODLRSolverMatrix<K>* matrix_;
    HODLR_Tree<HODLRSolverMatrix<K> >* solver_;
    VectorXd x_, yerr_;

    //
    // Compute the ``alpha`` vector for a given RHS.
    //
    VectorXd compute_alpha (MatrixXd y) {
        MatrixXd alpha(y.rows(), 1);
        solver_->solve(y, alpha);
        return alpha;
    };

};

};

#endif
