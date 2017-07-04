#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>

#include "george/hodlr.h"
#include "george/kernels.h"
#include "george/parser.h"
#include "george/exceptions.h"

namespace py = pybind11;

class SolverMatrix {
  public:
    SolverMatrix (george::kernels::Kernel* kernel)
      : kernel_(kernel) {};
    void set_input_coordinates (Eigen::MatrixXd x) {
      if (x.cols() != kernel_->get_ndim()) {
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
    Eigen::MatrixXd t_;
};

class Solver {
public:

  Solver (py::object& kernel_spec, int min_size = 100, double tol = 0.1, int seed = 0)
    : tol_(tol)
    , min_size_(min_size)
    , seed_(seed)
  {
    solver_ = NULL;
    kernel_ = george::parse_kernel_spec(kernel_spec);
    matrix_ = new SolverMatrix(kernel_);
    computed_ = 0;
  };
  ~Solver () {
    if (solver_ != NULL) delete solver_;
    delete matrix_;
    delete kernel_;
  };

  int get_status () const { return 0; };
  int get_computed () const { return computed_; };
  double log_determinant () const { return log_det_; };

  int compute (const Eigen::MatrixXd& x, const Eigen::VectorXd& yerr) {

    // Random number generator for reproducibility
    std::random_device r;
    std::mt19937 random(r());
    random.seed(seed_);

    int n = x.rows();
    computed_ = 0;
    Eigen::VectorXd diag;
    diag.array() = yerr.array() * yerr.array();
    matrix_->set_input_coordinates(x);

    // Set up the solver object.
    if (solver_ != NULL) delete solver_;
    solver_ = new george::hodlr::Node<SolverMatrix> (
        diag, matrix_, 0, n, min_size_, tol_, random);
    solver_->compute();
    log_det_ = solver_->log_determinant();

    // Update the bookkeeping flags.
    computed_ = 1;
    size_ = n;
    return 0;
  };

  void apply_inverse (Eigen::MatrixXd& x) {
    solver_->solve(x);
  };

  int size () const { return size_; };

private:
  double log_det_, tol_;
  int min_size_, seed_, size_;
  int computed_;

  george::kernels::Kernel* kernel_;
  SolverMatrix* matrix_;
  george::hodlr::Node<SolverMatrix>* solver_;
};



PYBIND11_PLUGIN(hodlr) {
  typedef Eigen::MatrixXd matrix_t;
  typedef Eigen::VectorXd vector_t;

  py::module m("hodlr", R"delim(
Docs...
)delim");

  py::class_<Solver> solver(m, "HODLRSolver");
  solver.def(py::init<py::object, int, double, int>(),
      py::arg("kernel_spec"), py::arg("min_size") = 100, py::arg("tol") = 10, py::arg("seed") = 42);
  solver.def_property_readonly("computed", &Solver::get_computed);
  solver.def_property_readonly("log_determinant", &Solver::log_determinant);
  solver.def("compute", &Solver::compute);
  solver.def("apply_inverse", [](Solver& self, Eigen::MatrixXd& x, bool in_place = false){
    if (in_place) {
      self.apply_inverse(x);
      return x;
    }
    Eigen::MatrixXd alpha = x;
    self.apply_inverse(alpha);
    return alpha;
  }, py::arg("x"), py::arg("in_place") = false);

  solver.def("get_inverse", [](Solver& self){
    int n = self.size();
    Eigen::MatrixXd eye(n, n);
    eye.setIdentity();
    self.apply_inverse(eye);
    return eye;
  });

  return m.ptr();
}
