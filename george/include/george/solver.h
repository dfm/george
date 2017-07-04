#ifndef _GEORGE_SOLVER_H_
#define _GEORGE_SOLVER_H_

#include <cmath>
#include <random>
#include <Eigen/Dense>

#include "hodlr.h"
#include "constants.h"
#include "kernels.h"

namespace george {

// Eigen is column major and numpy is row major. Barf.
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor> > RowMajorMap;

class SolverMatrix {
  public:
    SolverMatrix (george::kernels::Kernel* kernel)
      : kernel_(kernel)
    {
      stride_ = kernel_->get_ndim();
    };
    void set_input_coordinates (const double* v) {
      t_ = v;
    };
    double get_value (const int i, const int j) {
      return kernel_->value(&(t_[i*stride_]), &(t_[j*stride_]));
    };

  private:
    george::kernels::Kernel* kernel_;
    int stride_;
    const double* t_;
};

class Solver {
public:

  Solver (george::kernels::Kernel* kernel, int min_size = 10, double tol = 1.234e-5)
    : tol_(tol)
    , min_size_(min_size)
    , kernel_(kernel)
  {
    solver_ = NULL;
    matrix_ = new SolverMatrix(kernel_);
    computed_ = 0;
  };
  ~Solver () {
    if (solver_ != NULL) delete solver_;
    delete matrix_;
  };

  int get_status () const { return 0; };
  int get_computed () const { return computed_; };
  double get_log_determinant () const { return logdet_; };

  //
  // Pre-compute and factorize the kernel matrix.
  //
  int compute (const int n, const double* x, const double* yerr, int seed) {

    std::random_device r;
    std::mt19937 random(r());
    random.seed(seed);

    // It's not computed until it's computed...
    computed_ = 0;

    // Compute the diagonal elements.
    Eigen::VectorXd diag(n);
    for (int i = 0; i < n; ++i) {
      diag[i] = yerr[i]*yerr[i];
    }

    // Set the time points for the kernel.
    matrix_->set_input_coordinates(x);

    // Set up the solver object.
    if (solver_ != NULL) delete solver_;

    solver_ = new hodlr::Node<SolverMatrix> (
        diag, matrix_, 0, n, min_size_, tol_, random);
    solver_->compute();
    logdet_ = solver_->log_determinant();

    // Update the bookkeeping flags.
    computed_ = 1;
    return 0;
  };

  void apply_inverse (const int n, const int nrhs,
      double* b, double* out) {
    int i, j;
    Eigen::MatrixXd b_vec = RowMajorMap(b, n, nrhs),
      alpha = b_vec;
    solver_->solve(alpha);
    for (i = 0; i < n; ++i)
      for (j = 0; j < nrhs; ++j)
        out[i*nrhs+j] = alpha(i, j);
  };

private:
  double logdet_, tol_;
  int min_size_;
  int computed_;

  kernels::Kernel* kernel_;
  SolverMatrix* matrix_;
  hodlr::Node<SolverMatrix>* solver_;
};

};

#endif
