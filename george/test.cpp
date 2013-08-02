#include <iostream>
#include <Eigen/Dense>
#include "george.h"

#define NDATA 10

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix;
using George::GaussianProcess;
using George::IsotropicGaussianKernel;

int main ()
{
    int i;
    MatrixXd x = 10 * Matrix<double, NDATA, 1>::Random();
    VectorXd y(NDATA), yerr = 0.1 * VectorXd::Zero(NDATA), pars(2);

    pars[0] = 1.0;
    pars[1] = 1.0;

    IsotropicGaussianKernel kernel(pars);

    for (i = 0; i < NDATA; ++i) y(i) = cos(x(i));

    std::cout << y.rows() << std::endl;

    GaussianProcess gp(kernel);
    gp.compute(x, yerr);

    double l0 = gp.lnlikelihood(y), eps = 1e-4;
    VectorXd g0 = gp.gradlnlikelihood(y), g(2);

    for (i = 0; i < 2; ++i) {
        pars[i] += eps;
        kernel.set_pars(pars);
        gp.set_kernel(kernel);

        gp.compute(x, yerr);
        g[i] = (gp.lnlikelihood(y) - l0) / eps;
        pars[i] -= eps;
    }

    std::cout << g0 << std::endl;
    std::cout << g << std::endl;
    std::cout << g0 - g << std::endl;

    return 0;
}
