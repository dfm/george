#include <iomanip>
#include <iostream>
#include <ctime>

#include "george.h"

using Eigen::VectorXd;
using george::Kernel;
using george::SparseKernel;
using george::SparseSolver;
using george::HODLRSolver;

int main ()
{
    clock_t start, end;
    std::cout << std::setprecision(16);
    srand (123);

    // Generate some fake data.
    int N = 1000;
    VectorXd t(N), yerr(N), y(N);
    for (int i = 0; i < N; ++i) {
        t[i] = i * 1.0 / 60.0 / 24.;
        yerr[i] = 1e-2;
        y[i] = 1e-12 * (rand() - 0.5);
    }

    // Test the sparse solver.
    SparseKernel* kernel = new SparseKernel (1e-4, 1.0, 3.0);
    SparseSolver<Kernel> sparse_solver (kernel);
    if (sparse_solver.get_status()) {
        std::cerr << "Solver setup failed\n";
        delete kernel;
        return sparse_solver.get_status();
    }

    start = clock();
    sparse_solver.compute(t, yerr);
    end = clock();
    std::cout << "Sparse compute takes: " << double(end-start) / double(CLOCKS_PER_SEC) << std::endl;

    if (sparse_solver.get_status()) {
        std::cerr << "Compute failed\n";
        delete kernel;
        return sparse_solver.get_status();
    }

    start = clock();
    double ll = sparse_solver.log_likelihood (y);
    end = clock();
    std::cout << "Sparse log-like takes: " << double(end-start) / double(CLOCKS_PER_SEC) << std::endl;

    std::cout << ll << std::endl;
    std::cout << sparse_solver.get_status() << std::endl;

    // Test the HODLR solver.
    HODLRSolver<Kernel> hodlr_solver (kernel);

    start = clock();
    hodlr_solver.compute(t, yerr);
    end = clock();
    std::cout << "HODLR compute takes: " << double(end-start) / double(CLOCKS_PER_SEC) << std::endl;

    // Compute the log likelihood.
    start = clock();
    ll = hodlr_solver.log_likelihood (y);
    end = clock();
    std::cout << "HODLR log-like takes: " << double(end-start) / double(CLOCKS_PER_SEC) << std::endl;

    std::cout << ll << std::endl;
    std::cout << hodlr_solver.get_status() << std::endl;

    delete kernel;
    return 0;
}
