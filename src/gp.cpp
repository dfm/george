#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>


using namespace Eigen;
using namespace std;


template <class T>
T gaussianKernel(T x1, T x2, T *pars, int npars)
{
    T d = x1 - x2;
    d *= d / pars[1];
    return pars[0] * exp(-0.5 * d);
}


template <class T>
SparseMatrix<T> buildK(Matrix<T, Dynamic, 1> x1,
                       Matrix<T, Dynamic, 1> x2,
                       T (*k) (T, T, T*, int),
                       T *pars, int npars)
{
    int i, j;
    int N1 = x1.rows(), N2 = x2.rows();

    SparseMatrix<T> m(N1, N2);

    typedef Triplet<T> triplet;
    vector<triplet> triplets;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            T val = k(x1[i], x2[j], pars, npars);

            // MAGIC: sparse zero cutoff.
            if (val > 1e-5)
                triplets.push_back(triplet(i, j, val));
        }
    }

    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();

    return m;
}


int main()
{
    const int ndata = 5;
    VectorXd x(ndata);
    x << -4.0, -3.6, -0.2, 0.5, 4.6;
    VectorXd y(ndata);
    y << -2.0, 5.6, 2.1, -0.5, 3.0;

    const int ntarget = 100;
    VectorXd target = VectorXd::LinSpaced(Sequential, ntarget, -5.0, 5.0);

    const int npars = 2;
    double pars[] = {1.0, 1.0};

    SparseMatrix<double> m = buildK<double>(x, x,
                                            gaussianKernel<double>,
                                            pars, npars);

    // for (int k = 0; k < m.outerSize(); ++k) {
    //     for (SparseMatrix<double>::InnerIterator it(m, k); it; ++it) {
    //         it.value();
    //         it.row();   // row index
    //         it.col();   // col index (here it is equal to k)
    //         it.index(); // inner index, here it is equal to it.row()
    //     }
    // }

    SimplicialLLT<SparseMatrix<double> > L(m);
    VectorXd alpha = L.solve(y);

    SparseMatrix<double> kstar = buildK<double>(target, x,
                                            gaussianKernel<double>,
                                            pars, npars);
    cout << kstar * alpha << endl;

    return 0;
}
