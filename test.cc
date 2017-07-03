#include <random>
#include <iostream>

#include <Eigen/Core>

#include "hodlr.h"

class MyKernel {
public:
  double get_value (int i, int j) {
    double tau = i - j;
    return exp(-0.5 * tau * tau) * cos(2 * M_PI * tau);
  };
};

int main () {
  std::random_device r;
  std::mt19937 random(r());
  random.seed(42);

  int N = 301;

  MyKernel kernel;
  Eigen::VectorXd diag(N);
  diag.setOnes();

  george::hodlr::Node<MyKernel> solver(diag, &kernel, 0, N, 10, 1.234e-5, random);

  Eigen::MatrixXd K = solver.get_exact_matrix();
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(N, 5), b,
    alpha = K.ldlt().solve(x);

  solver.compute();

  b = x;
  solver.solve(b);

  std::cout << std::scientific;
  std::cout << "solve error: " << (b - alpha).cwiseAbs().maxCoeff() << std::endl;
  std::cout << "dot solve error: " << (solver.dot_solve(x) - x.transpose() * alpha).cwiseAbs().maxCoeff() << std::endl;
  std::cout << "logdet error: " << solver.log_determinant() - log(K.determinant()) << std::endl;

  return 0;
}
