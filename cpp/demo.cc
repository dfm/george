#include <iostream>

#include "george.h"

using Eigen::VectorXd;
using george::SparseKernel;
using george::SparseSolver;
using george::HODLRSolver;

int main ()
{
    // Generate some fake data.
    int N = 500;
    VectorXd t(N), yerr(N), y(N);
    for (int i = 0; i < N; ++i) {
        t[i] = i * 0.5 / 24.;
        yerr[i] = 1e-4;
        y[i] = 1e-12 * (rand() - 0.5);
    }

    // Test the sparse solver.
    SparseKernel* kernel = new SparseKernel (1e-4, 1.0, 3.0);
    SparseSolver<SparseKernel> sparse_solver (kernel);
    if (sparse_solver.get_status()) {
        std::cerr << "Solver setup failed\n";
        delete kernel;
        return sparse_solver.get_status();
    }

    sparse_solver.compute(t, yerr);
    if (sparse_solver.get_status()) {
        std::cerr << "Compute failed\n";
        delete kernel;
        return sparse_solver.get_status();
    }

    std::cout << sparse_solver.log_likelihood (y) << std::endl;
    std::cout << sparse_solver.get_status() << std::endl;

    // Test the HODLR solver.
    HODLRSolver<SparseKernel> hodlr_solver (kernel);

    delete kernel;
    return 0;
}
