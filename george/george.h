#ifndef _GEORGE_H_
#define _GEORGE_H_

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/AutoDiff>

#define TWOLNPI 1.8378770664093453

using std::vector;
using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Dynamic;
using Eigen::LDLT;
using Eigen::Success;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Eigen::SimplicialLDLT;
using Eigen::AutoDiffScalar;

namespace George {

    class Kernel {

    public:
        Kernel () {};
        Kernel (VectorXd pars) {
            pars_ = pars;
        };
        virtual ~Kernel () { };

        int npars () const { return pars_.rows(); };
        VectorXd pars () const { return pars_; };
        void set_pars (VectorXd pars) { pars_ = pars; };

        virtual double evaluate (VectorXd x1, VectorXd x2) {
            return 0.0;
        };

        virtual void gradient (VectorXd x1, VectorXd x2, double *value,
                               VectorXd *grad) {
            *value = 0.0;
            *grad = VectorXd::Zero(pars_.rows());
        };

    protected:
        VectorXd pars_;

    };

    class IsotropicGaussianKernel : public Kernel {

    public:

        IsotropicGaussianKernel() {};
        IsotropicGaussianKernel(VectorXd pars) : Kernel (pars) {};

        template <typename T>
        T compute (VectorXd x1, VectorXd x2, Matrix<T, Dynamic, 1> p) {
            int jp1 = int(0.5*x1.rows())+2;
            VectorXd d = x1 - x2;
            double chi2 = d.dot(d);
            T r = sqrt(chi2) / sqrt(p[2] * p[2]),
              omr = 1.0 - r,
              f;

            // Compute the compact envelope function.
            if (r >= 1.0) return 0.0;
            f = pow(omr, jp1)*(jp1*r + 1.0);

            // Compute the squared-exp covariance.
            return p[0] * p[0] * exp(-0.5 * chi2 / p[1] * p[1]) * f;
        };

        double evaluate (VectorXd x1, VectorXd x2) {
            return compute<double>(x1, x2, pars_);
        };

        void gradient (VectorXd x1, VectorXd x2, double *value, VectorXd *grad) {
            typedef Matrix<double, Dynamic, 1> derivative_t;
            typedef AutoDiffScalar<derivative_t> scalar_t;
            typedef Matrix<scalar_t, Dynamic, 1> input_t;

            int i, j, N = npars();
            input_t p(N);
            p = pars_.cast<scalar_t>();

            for (i = 0; i < N; ++i) {
                p(i).derivatives().resize(N);
                for (j = 0; j < N; ++j) p(i).derivatives()(j) = pars_(j);
                p(i).derivatives()(i) += 1.0;
            }

            scalar_t y = compute<scalar_t>(x1, x2, p);

            *grad = y.derivatives();
            if (!grad->size()) *grad = VectorXd::Zero(pars_.rows());

            *value = y.value();
        };

    };

    class GaussianProcess {

    private:
        Kernel *kernel_;
        int info_;
        bool computed_;
        MatrixXd x_;
        SimplicialLDLT<SparseMatrix<double> > *L_;

    public:
        GaussianProcess () {
            info_ = 0;
            computed_ = false;
            L_ = new SimplicialLDLT<SparseMatrix<double> > ();
            kernel_ = new Kernel ();
        };
        ~GaussianProcess () {
            delete L_;
            delete kernel_;
        };

        Kernel *kernel () const { return kernel_; };
        void set_kernel (Kernel *k) {
            delete kernel_;
            kernel_ = k;
            computed_ = false;
        };

        int info () const { return info_; };
        int computed () const { return computed_; };
        int nsamples () const { return x_.rows(); };

        int compute (MatrixXd x, VectorXd yerr)
        {
            typedef Triplet<double> T;

            int i, j, nsamples = x.rows();
            double value;
            SparseMatrix<double> Kxx(nsamples, nsamples);
            vector<T> entries;
            x_ = x;

            // Build the sparse covariance matrix.
            for (i = 0; i < nsamples; ++i) {
                entries.push_back(T(i, i,
                                    kernel_->evaluate(x.row(i), x.row(i))
                                    + yerr[i]*yerr[i]));
                for (j = i + 1; j < nsamples; ++j) {
                    value = kernel_->evaluate(x.row(i), x.row(j));
                    if (value > 0) entries.push_back(T(j, i, value));
                }
            }
            Kxx.setFromTriplets(entries.begin(), entries.end());
            Kxx.makeCompressed();

            // Factorize the covariance.
            delete L_;
            L_ = new SimplicialLDLT<SparseMatrix<double> > (Kxx);
            if (L_->info() != Success) return -1;

            computed_ = true;
            return 0;
        };

        double lnlikelihood (VectorXd y)
        {
            double logdet;
            VectorXd alpha;

            if (!computed_ || y.rows() != x_.rows())
                return -INFINITY;

            alpha = L_->solve(y);
            if (L_->info() != Success)
                return -INFINITY;

            logdet = log(L_->vectorD().array()).sum();
            return -0.5 * (y.transpose() * alpha + logdet + y.rows() * TWOLNPI);
        };

        VectorXd gradlnlikelihood (VectorXd y)
        {
            int i, j, k, nsamples = y.rows(), npars = kernel_->npars();
            double tmp;
            VectorXd grad(npars), alpha;
            vector<MatrixXd> dkdt(npars);

            if (!computed_ || y.rows() != x_.rows()) {
                info_ = -1;
                return grad;
            }

            alpha = L_->solve(y);
            if (L_->info() != Success) {
                info_ = -2;
                return grad;
            }

            // Initialize the gradient matrices.
            for (i = 0; i < npars; ++i) dkdt[i] = MatrixXd(nsamples, nsamples);

            // Compute the gradient matrices.
            for (i = 0; i < nsamples; ++i)
                for (j = i; j < nsamples; ++j) {
                    kernel_->gradient(x_.row(i), x_.row(j), &tmp, &grad);
                    for (k = 0; k < npars; ++k) {
                        dkdt[k](i, j) = grad(k);
                        if (j > i) dkdt[k](j, i) = grad(k);
                    }
                }

            // Compute the gradient.
            for (k = 0; k < npars; ++k) {
                grad(k) = L_->solve(dkdt[k]).trace();
                for (i = 0; i < nsamples; ++i)
                    grad(k) -= alpha(i) * alpha.dot(dkdt[k].row(i));
                grad(k) *= -0.5;
            }

            return grad;
        };

        int predict (VectorXd y, MatrixXd x, VectorXd *mu, MatrixXd *cov)
        {
            int i, j, ntest = x.rows(), nsamples = y.rows();
            double value;
            MatrixXd Kxs(nsamples, ntest);
            VectorXd alpha, ytest;

            if (!computed_ || nsamples != x_.rows()) return -1;

            // Build the kernel matrices.
            for (i = 0; i < nsamples; ++i)
                for (j = 0; j < ntest; ++j)
                    Kxs(i, j) = kernel_->evaluate(x_.row(i), x.row(j));

            (*cov).resize(ntest, ntest);
            for (i = 0; i < ntest; ++i) {
                (*cov)(i, i) = kernel_->evaluate(x.row(i), x.row(i));
                for (j = i + 1; j < ntest; ++j) {
                    value = kernel_->evaluate(x.row(i), x.row(j));
                    (*cov)(i, j) = value;
                    (*cov)(j, i) = value;
                }
            }

            alpha = L_->solve(y);
            if (L_->info() != Success) return -2;

            // Compute and copy the predictive mean vector.
            *mu = Kxs.transpose() * alpha;

            // Compute the predictive covariance matrix.
            *cov -= Kxs.transpose() * L_->solve(Kxs);
            if (L_->info() != Success) return -3;

            return 0;
        };

    };
}

#endif
// /_GEORGE_H_
