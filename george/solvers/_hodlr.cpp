#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>

#include "george/hodlr.h"
#include "george/kernels.h"
#include "george/parser.h"
#include "george/exceptions.h"

namespace py = pybind11;

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class SolverMatrix {
  public:
    SolverMatrix (george::kernels::Kernel* kernel)
      : kernel_(kernel) {};
    void set_input_coordinates (RowMatrixXd x) {
      if (size_t(x.cols()) != kernel_->get_ndim()) {
        throw george::dimension_mismatch();
      }
      t_ = x;
    };
    double get_value (const int i, const int j) {
      if (i < 0 || i >= t_.rows() || j < 0 || j >= t_.rows()) {
        throw std::out_of_range("attempting to index outside of the dimension of the input coordinates");
      }
      return kernel_->value(t_.row(i).data(), t_.row(j).data());
    };

  private:
    george::kernels::Kernel* kernel_;
    RowMatrixXd t_;
};

class Solver {
public:

  Solver () {
    solver_ = NULL;
    kernel_ = NULL;
    matrix_ = NULL;
    computed_ = 0;
  };
  ~Solver () {
    if (solver_ != NULL) delete solver_;
    if (matrix_ != NULL) delete matrix_;
    if (kernel_ != NULL) delete kernel_;
  };

  int get_status () const { return 0; };
  int get_computed () const { return computed_; };
  double log_determinant () const { return log_det_; };

  int compute (
      const py::object& kernel_spec,
      const py::array_t<double>& x,
      const py::array_t<double>& yerr,
      int min_size = 100, double tol = 0.1, int seed = 0
  ) {
    computed_ = 0;
    kernel_ = george::parse_kernel_spec(kernel_spec);
    matrix_ = new SolverMatrix(kernel_);

    // Random number generator for reproducibility
    std::random_device r;
    std::mt19937 random(r());
    random.seed(seed);

    // Extract the data from the numpy arrays
    py::detail::unchecked_reference<double, 2L> x_p = x.unchecked<2>();
    py::detail::unchecked_reference<double, 1L> yerr_p = yerr.unchecked<1>();
    size_t n = x_p.shape(0), ndim = x_p.shape(1);
    RowMatrixXd X(n, ndim);
    Eigen::VectorXd diag(n);
    for (size_t i = 0; i < n; ++i) {
      diag(i) = yerr_p(i) * yerr_p(i);
      for (size_t j = 0; j < ndim; ++j) X(i, j) = x_p(i, j);
    }

    matrix_->set_input_coordinates(X);

    // Set up the solver object.
    if (solver_ != NULL) delete solver_;
    solver_ = new george::hodlr::Node<SolverMatrix> (
        diag, matrix_, 0, n, min_size, tol, random);
    solver_->compute();
    log_det_ = solver_->log_determinant();

    // Update the bookkeeping flags.
    computed_ = 1;
    size_ = n;
    return 0;
  };

  template <typename Derived>
  void apply_inverse (Eigen::MatrixBase<Derived>& x) {
    if (!computed_) throw george::not_computed();
    solver_->solve(x);
  };

  int size () const { return size_; };

private:
  double log_det_;
  int size_;
  int computed_;

  george::kernels::Kernel* kernel_;
  SolverMatrix* matrix_;
  george::hodlr::Node<SolverMatrix>* solver_;
};


PYBIND11_MODULE(_hodlr, m) {
  py::class_<Solver> solver(m, "HODLRSolver", R"delim(
A solver using `Sivaram Amambikasaran's HODLR algorithm
<http://arxiv.org/abs/1403.6015>`_ to approximately solve the GP linear
algebra in :math:`\mathcal{O}(N\,\log^2 N)`.

)delim");
  solver.def(py::init());
  solver.def_property_readonly("computed", &Solver::get_computed);
  solver.def_property_readonly("log_determinant", &Solver::log_determinant);
  solver.def("compute", &Solver::compute, R"delim(
Compute and factorize the covariance matrix.

Args:
    kernel (george.kernels.Kernel): A subclass of :class:`Kernel` specifying
        the kernel function.
    x (ndarray[nsamples, ndim]): The independent coordinates of the data
        points.
    yerr (ndarray[nsamples]): The Gaussian uncertainties on the data points at
        coordinates ``x``. These values will be added in quadrature to the
        diagonal of the covariance matrix.
    min_size (Optional[int]): The block size where the solver switches to a
        general direct factorization algorithm. This can be tuned for platform
        and problem specific performance and accuracy. As a general rule,
        larger values will be more accurate and slower, but there is some
        overhead for very small values, so we recommend choosing values in the
        hundreds. (default: ``100``)
    tol (Optional[float]): The precision tolerance for the low-rank
        approximation. This value is used as an approximate limit on the
        Frobenius norm between the low-rank approximation and the true matrix
        when reconstructing the off-diagonal blocks. Smaller values of ``tol``
        will generally give more accurate results with higher computational
        cost. (default: ``0.1``)
    seed (Optional[int]): The low-rank approximation method within the HODLR
        algorithm is not deterministic and, without a fixed seed, the method
        can give different results for the same matrix. Therefore, we require
        that the user provide a seed for the random number generator.
        (default: ``42``, obviously)
)delim",
    py::arg("kernel_spec"), py::arg("x"), py::arg("yerr"), py::arg("min_size") = 100, py::arg("tol") = 0.1, py::arg("seed") = 42
  );
  solver.def("apply_inverse", [](Solver& self, Eigen::MatrixXd& x, bool in_place = false){
    if (in_place) {
      self.apply_inverse(x);
      return x;
    }
    Eigen::MatrixXd alpha = x;
    self.apply_inverse(alpha);
    return alpha;
  }, py::arg("x"), py::arg("in_place") = false, R"delim(
Apply the inverse of the covariance matrix to the input by solving

.. math::

    K\,x = y

Args:
    y (ndarray[nsamples] or ndadrray[nsamples, nrhs]): The vector or matrix
        :math:`y`.
    in_place (Optional[bool]): Should the data in ``y`` be overwritten with
        the result :math:`x`? (default: ``False``)
)delim");

  solver.def("dot_solve", [](Solver& self, const Eigen::VectorXd& x){
    Eigen::VectorXd alpha = x;
    self.apply_inverse(alpha);
    return double(x.transpose() * alpha);
  }, R"delim(
Compute the inner product of a vector with the inverse of the covariance
matrix applied to itself:

.. math::

    y\,K^{-1}\,y

Args:
    y (ndarray[nsamples]): The vector :math:`y`.
)delim");

  solver.def("get_inverse", [](Solver& self){
    int n = self.size();
    Eigen::MatrixXd eye(n, n);
    eye.setIdentity();
    self.apply_inverse(eye);
    return eye;
  }, R"delim(
Get the dense inverse covariance matrix. This is used for computing gradients,
but it is not recommended in general.
)delim");
}
