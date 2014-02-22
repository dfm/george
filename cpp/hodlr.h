#ifndef _GEORGE_HODLR_SOLVER_H_
#define _GEORGE_HODLR_SOLVER_H_

#include <cmath>
#include <Eigen/Dense>
#include <HODLR_Tree.hpp>
#include <HODLR_Matrix.hpp>

#include "constants.h"

using Eigen::VectorXd;

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
    HODLRSolver (K* kernel, unsigned nLeaf = 50, double tol = 1e-12)
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

        return 0;
    };

private:

    double logdet_, tol_;
    unsigned nleaf_;
    int status_, computed_;
    K* kernel_;
    HODLRSolverMatrix<K>* matrix_;
    HODLR_Tree<HODLRSolverMatrix<K> > solver_;

};

};

#endif
