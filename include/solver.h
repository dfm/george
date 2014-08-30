#ifndef _GEORGE_SOLVER_H_
#define _GEORGE_SOLVER_H_

#include <cmath>
#include <Eigen/Dense>
#include <HODLR_Tree.hpp>
#include <HODLR_Matrix.hpp>

#include "constants.h"
#include "kernels.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using george::kernels::Kernel;

namespace george {

class HODLRSolverMatrix : public HODLR_Matrix {
public:
    HODLRSolverMatrix (Kernel* kernel)
        : kernel_(kernel)
    {
        stride_ = kernel_->get_ndim();
    };
    void set_values (double* v) {
        t_ = v;
    };
    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return kernel_->value(&(t_[i*stride_]), &(t_[j*stride_]));
    };

private:
    Kernel* kernel_;
    unsigned int stride_;
    double* t_;
};

class HODLRSolver {

public:

    //
    // Allocation and deallocation.
    //
    HODLRSolver (Kernel* kernel, unsigned nLeaf = 10, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), kernel_(kernel)
    {
        matrix_ = new HODLRSolverMatrix(kernel_);
        solver_ = NULL;
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~HODLRSolver () {
        if (solver_ != NULL) delete solver_;
        delete matrix_;
    };

    //
    // Accessors.
    //
    int get_status () const { return status_; };
    int get_computed () const { return computed_; };
    double get_log_determinant () const { return logdet_; };

    //
    // Pre-compute and factorize the kernel matrix.
    //
    int compute (const unsigned int n, const double* x, const double* yerr,
                 unsigned int seed) {
        unsigned int ndim = kernel_->get_ndim();

        // It's not computed until it's computed...
        computed_ = 0;

        // Compute the diagonal elements.
        VectorXd diag(n);
        for (int i = 0; i < n; ++i) {
            diag[i] = yerr[i]*yerr[i];
            diag[i] += kernel_->value(&(x[i*ndim]), &(x[i*ndim]));
        }

        // Set the time points for the kernel.
        matrix_->set_values(x);

        // Set up the solver object.
        if (solver_ != NULL) delete solver_;
        solver_ = new HODLR_Tree<HODLRSolverMatrix> (matrix_, n, nleaf_);
        solver_->assemble_Matrix(diag, tol_, 's', seed);

        // Factorize the matrix.
        solver_->compute_Factor();

        // Extract the log-determinant.
        solver_->compute_Determinant(logdet_);

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

private:
    double logdet_, tol_;
    unsigned nleaf_;
    int status_, computed_;
    Kernel* kernel_;
    HODLRSolverMatrix* matrix_;
    HODLR_Tree<HODLRSolverMatrix>* solver_;
    MatrixXd x_;
};

};

#endif
