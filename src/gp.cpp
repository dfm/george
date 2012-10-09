#include "gp.h"


using namespace Eigen;


int GaussianProcess::fit(MatrixXd x, VectorXd y, VectorXd yerr)
{
    x_ = x;
    y_ = y;
    yerr_ = yerr;

    ndim_ = x.cols();
    nsamples_ = x.rows();

    // Sanity check the dimensions
    if (y.rows() != nsamples_) return 1;
    if (yerr.rows() != nsamples_) return 2;

    // Build the base kernel
    Kxx_ = K(x, x);

    // Add in the noise
    for (int n = 0; n < nsamples_; ++n)
        Kxx_(n, n) += yerr(n) * yerr(n);

    // Compute the decomposition of K(X, X)
    L_ = LDLT<MatrixXd>(Kxx_);
    if (L_.info() != Success)
        return -1;

    alpha_ = L_.solve(y);
    if (L_.info() != Success)
        return -2;

    return 0;
}


double GaussianProcess::evaluate()
{
    double logdet = log(L_.vectorD().array()).sum();
    return -0.5 * (y_.transpose() * alpha_ + logdet + nsamples_ * l2pi_);
}


int GaussianProcess::predict(MatrixXd x, VectorXd *mean, MatrixXd *cov)
{
    MatrixXd kstar = K(x_, x);
    *mean = kstar.transpose() * alpha_;

    *cov = K(x, x);
    *cov -= kstar.transpose() * L_.solve(kstar);
    if (L_.info() != Success)
        return -1;

    return 0;
}


MatrixXd GaussianProcess::K(MatrixXd x1, MatrixXd x2)
{
    int i, j;
    int N1 = x1.rows(), N2 = x2.rows();
    MatrixXd m(N1, N2);

    for (i = 0; i < N1; i++)
        for (j = 0; j < N2; j++)
            m(i, j) = kernel_(x1.row(i), x2.row(j), pars_);

    return m;
}
