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

// Eigen is column major and numpy is row major. Barf.
typedef Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> > RowMajorMap;

class HODLRSolverMatrix : public HODLR_Matrix {
public:
    HODLRSolverMatrix (Kernel* kernel)
        : kernel_(kernel)
    {
        stride_ = kernel_->get_ndim();
    };
    void set_values (const double* v) {
        t_ = v;
    };
    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return kernel_->value(&(t_[i*stride_]), &(t_[j*stride_]));
    };

private:
    Kernel* kernel_;
    unsigned int stride_;
    const double* t_;
};

class Solver {

public:

    //
    // Allocation and deallocation.
    //
    Solver (Kernel* kernel, unsigned nLeaf = 10, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), kernel_(kernel)
    {
        matrix_ = new HODLRSolverMatrix(kernel_);
        solver_ = NULL;
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~Solver () {
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
        for (unsigned int i = 0; i < n; ++i) {
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

    void apply_inverse (const unsigned int n, const unsigned int nrhs,
                        double* b, double* out) {
        unsigned int i, j;
        MatrixXd b_vec = RowMajorMap(b, n, nrhs), alpha(n, nrhs);
        solver_->solve(b_vec, alpha);
        for (i = 0; i < n; ++i)
            for (j = 0; j < nrhs; ++j)
                out[i*nrhs+j] = alpha(i, j);
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
