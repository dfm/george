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
    MatrixXd x = Matrix<double, NDATA, 1>::Random();
    VectorXd y = x.col(0), yerr = 0.1 * VectorXd::Zero(NDATA);
    VectorXd pars(2);
    pars[0] = 1.0;
    pars[1] = 1.0;
    IsotropicGaussianKernel kernel(pars);

    std::cout << y.rows() << std::endl;

    GaussianProcess<IsotropicGaussianKernel> gp(kernel);
    gp.compute(x, yerr);
    std::cout << gp.lnlikelihood(y) << std::endl;
    std::cout << gp.gradlnlikelihood(y) << std::endl;

    return 0;
}
