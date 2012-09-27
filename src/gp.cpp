#include <iostream>
#include <Eigen/SparseCore>


using namespace Eigen;
using namespace std;


template <class T>
SparseMatrix<T> buildK(T a2, T l2, T chi2max, T *x1, T *x2, int N1, int N2)
{
    int i, j;

    SparseMatrix<T> m(N1, N2);

    typedef Triplet<T> triplet;
    vector<triplet> triplets;

    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            T d = x1[i] - x2[j];
            d *= d / l2;
            if (d < chi2max)
                triplets.push_back(triplet(i, j, a2 * exp(-0.5 * d)));
        }
    }

    m.setFromTriplets(triplets.begin(), triplets.end());
    m.makeCompressed();

    return m;
}


int main()
{
    int rows = 10, cols = 5;

    double x[5] = {-4.0, -3.6, -0.2, 0.5, 4.6};
    double y[5] = {-2.0, 5.6, 2.1, -0.5, 3.0};

    SparseMatrix<double> m = buildK<double>(1.0, 1.0, 25.0, x, x, 5, 5);

    for (int k = 0; k < m.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(m, k); it; ++it) {
            cout << it.value() << endl;
            it.row();   // row index
            it.col();   // col index (here it is equal to k)
            it.index(); // inner index, here it is equal to it.row()
        }
    }

    return 0;
}
