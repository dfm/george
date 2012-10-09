#include <iostream>
#include "gp.h"


using namespace Eigen;
using namespace std;


//
// Kernels.
//

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

//
// Algorithm.
//

MatrixXd buildK(MatrixXd x1, MatrixXd x2, VectorXd pars,
                double (*k) (VectorXd, VectorXd, VectorXd))
{
    int i, j;
    int N1 = x1.rows(), N2 = x2.rows();
    MatrixXd m(N1, N2);

    for (i = 0; i < N1; i++)
        for (j = 0; j < N2; j++)
            m(i, j) = k(x1.row(i), x2.row(j), pars);

    return m;
}

int evaluateGP(MatrixXd x, VectorXd y, VectorXd sigma, MatrixXd target,
               VectorXd pars,
               double (*kernel) (VectorXd, VectorXd, VectorXd),
               VectorXd *mean, MatrixXd *cov, double *loglike)
{
    int ndim = x.cols();
    int nsamples = x.rows();
    int ntarget = target.rows();

    if (y.rows() != nsamples) return 1;
    if (sigma.rows() != nsamples) return 2;
    if (target.cols() != ndim) return 3;

    /* Build the base kernel */
    MatrixXd Kxx = buildK(x, x, pars, kernel);

    /* Add in the noise */
    for (int n = 0; n < nsamples; ++n)
        Kxx(n, n) += sigma(n) * sigma(n);

    /* Find alpha */
    LDLT<MatrixXd> L(Kxx);
    if (L.info() != Success)
        return -1;

    VectorXd alpha = L.solve(y);
    if (L.info() != Success)
        return -2;

    /* Compute the mean */
    MatrixXd kstar = buildK(x, target, pars, kernel);
    *mean = kstar.transpose() * alpha;

    /* Compute the covariance */
    *cov = buildK(target, target, pars, kernel);
    *cov -= kstar.transpose() * L.solve(kstar);

    /* Compute the log-likelihood */
    double logdet = log(L.vectorD().array()).sum();
    *loglike = -0.5 * (y.transpose() * alpha + logdet
                                                + nsamples * log(2 * M_PI));

    return 0;
}
