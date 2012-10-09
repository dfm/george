#include "kernels.h"

using namespace Eigen;

double isotropicKernel(VectorXd x1, VectorXd x2, VectorXd pars)
{
    VectorXd d = x1 - x2;
    double result = pars[0] * exp(-0.5 * d.dot(d) / pars[1]);
    return result;
}

double diagonalKernel(VectorXd x1, VectorXd x2, VectorXd pars)
{
    VectorXd d = x1 - x2;
    double result = 0.0;
    for (int i = 0; i < d.rows(); ++i)
        result += d[i] * d[i] / pars[i + 1];
    return pars[0] * exp(-0.5 * result);
}
