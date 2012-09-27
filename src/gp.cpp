#include "gp.h"


using namespace Eigen;
using namespace std;


template <class T>
T gaussianKernel(T x1, T x2, Matrix<T, Dynamic, 1>pars)
{
    T d = x1 - x2;
    d *= d / pars[1];
    return pars[0] * exp(-0.5 * d);
}


template <class T>
SparseMatrix<T> buildK(Matrix<T, Dynamic, 1> x1,
                       Matrix<T, Dynamic, 1> x2,
                       Matrix<T, Dynamic, 1> pars,
                       T (*k) (T, T, Matrix<T, Dynamic, 1>))
{
    int i, j;
    int N1 = x1.rows(), N2 = x2.rows();

    SparseMatrix<T> m(N1, N2);

    typedef Triplet<T> triplet;
    vector<triplet> triplets;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            T val = k(x1[i], x2[j], pars);

            // MAGIC: sparse zero cutoff.
            if (val > 1e-5)
                triplets.push_back(triplet(i, j, val));
        }
    }

    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();

    return m;
}


int evaluateGP(VectorXd x, VectorXd y, VectorXd sigma, VectorXd target,
               VectorXd *mean, VectorXd *variance, double *loglike)
{
    VectorXd pars(2);
    pars << 1.0, 1.0;

    /* Build the base kernel */
    SparseMatrix<double> Kxx = buildK<double>(x, x, pars,
                                              gaussianKernel<double>);

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
    SparseMatrix<double> kstar = buildK<double>(x, target, pars,
                                                gaussianKernel<double>);
    *mean = kstar.transpose() * alpha;

    /* Compute the variance */
    (*variance).resize(target.rows());
    for (int i = 0; i < kstar.outerSize(); ++i) {
        VectorXd k = VectorXd::Zero(x.rows());
        for (SparseMatrix<double>::InnerIterator it(kstar, i); it; ++it)
            k[it.row()] = it.value();
        (*variance)[i] = gaussianKernel<double>(target[i], target[i], pars)
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
/*     const int ndata = 5; */

/*     VectorXd x(ndata); */
/*     x << -4.0, -3.6, -0.2, 0.5, 4.6; */
/*     VectorXd y(ndata); */
/*     y << -2.0, 5.6, 2.1, -0.5, 3.0; */
/*     VectorXd sigma(ndata); */
/*     y << 1.0, 0.76, 0.5, 0.6, 1.3; */

/*     VectorXd target = VectorXd::LinSpaced(Sequential, 100, -5.0, 5.0); */
/*     VectorXd mean(1), variance(1); */
/*     double loglike; */

/*     evaluateGP(x, y, sigma, target, &mean, &variance, &loglike); */

/*     cout << loglike << endl; */

/*     return 0; */
/* } */
