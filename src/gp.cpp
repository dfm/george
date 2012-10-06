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

SparseMatrix<double> buildK(MatrixXd x1, MatrixXd x2, VectorXd pars,
                            double sparsetol,
                            double (*k) (VectorXd, VectorXd, VectorXd))
{
    int i, j;
    int N1 = x1.rows(), N2 = x2.rows();

    SparseMatrix<double> m(N1, N2);

    typedef Triplet<double> triplet;
    vector<triplet> triplets;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            double val = k(x1.row(i), x2.row(j), pars);
            if (val > sparsetol)
                triplets.push_back(triplet(i, j, val));
        }
    }

    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();

    return m;
}

int evaluateGP(MatrixXd x, VectorXd y, VectorXd sigma, MatrixXd target,
               VectorXd pars,
               double (*kernel) (VectorXd, VectorXd, VectorXd),
               VectorXd *mean, VectorXd *variance, double *loglike,
               double sparsetol)
{
    /* Build the base kernel */
    SparseMatrix<double> Kxx = buildK(x, x, pars, sparsetol, kernel);

    /* Add in the noise */
    for (int n = 0; n < x.rows(); ++n)
        Kxx.coeffRef(n, n) += sigma[n] * sigma[n];

    /* Find alpha */
    SimplicialLLT<SparseMatrix<double> > L(Kxx);
    if (L.info() != Success)
        return -1;

    VectorXd alpha = L.solve(y);
    if (L.info() != Success)
        return -2;

    /* Compute the mean */
    SparseMatrix<double> kstar = buildK(x, target, pars, sparsetol, kernel);
    *mean = kstar.transpose() * alpha;

    /* Compute the variance */
    (*variance).resize(target.rows());
    for (int i = 0; i < kstar.outerSize(); ++i) {
        VectorXd k = VectorXd::Zero(x.rows());
        for (SparseMatrix<double>::InnerIterator it(kstar, i); it; ++it)
            k[it.row()] = it.value();
        (*variance)[i] = kernel(target.row(i), target.row(i), pars)
                                    - k.transpose() * L.solve(k);
        if (L.info() != Success)
            return -3;
    }

    /* Compute the log-likelihood */
    *loglike = -0.5 * (y.transpose() * alpha + log(L.determinant())
                                + x.rows() * log(2 * M_PI));

    return 0;
}


/* int main() */
/* { */
/*     const int ndata = 5, ntarget = 100, ndim = 1; */

/*     MatrixXd x(ndata, ndim); */
/*     x << -4.0, -3.6, -0.2, 0.5, 4.6; */
/*     VectorXd y(ndata); */
/*     y << -2.0, 5.6, 2.1, -0.5, 3.0; */
/*     VectorXd sigma(ndata); */
/*     y << 1.0, 0.76, 0.5, 0.6, 1.3; */

/*     double mn = -5.0, mx = 5.0; */
/*     MatrixXd target(ntarget, ndim); */
/*     for (int i = 0; i < ntarget; ++i) */
/*         target(i) = double(i) / ntarget * (mx - mn) + mn; */

/*     VectorXd mean(1), variance(1); */
/*     double loglike; */

/*     VectorXd pars(2); */
/*     pars << 1.0, 2.0; */

/*     evaluateGP(x, y, sigma, target, pars, isotropicKernel, */
/*                &mean, &variance, &loglike); */

/*     cout << loglike << endl; */

/*     return 0; */
/* } */
