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

  Eigen::MatrixXd x = Eigen::MatrixXd::Random(N, 5), x0 = x;

  george::hodlr::Node<MyKernel> solver(diag, &kernel, 0, N, 10, 1.234e-5, random);
  solver.compute();
  solver.apply_inverse(x);

  Eigen::MatrixXd K = solver.get_exact_matrix();

  std::cout << std::scientific;
  std::cout << "solve error: " << (x - K.ldlt().solve(x0)).cwiseAbs().maxCoeff() << std::endl;
  std::cout << "logdet error: " << solver.log_determinant() - log(K.determinant()) << std::endl;

  return 0;
}
