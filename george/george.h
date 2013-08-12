#ifndef _GEORGE_H_
#define _GEORGE_H_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#define TWOLNPI 1.8378770664093453

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::LDLT;
using Eigen::Success;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Eigen::SimplicialLDLT;

namespace George {

    class Kernel {

    public:
        Kernel () {};
        Kernel (VectorXd pars) {
            pars_ = pars;
        };

        int npars () const { return pars_.rows(); };
        VectorXd pars () const { return pars_; };
        void set_pars (VectorXd pars) { pars_ = pars; };

        virtual double evaluate (VectorXd x1, VectorXd x2) {
            return 0.0;
        };
        virtual VectorXd gradient (VectorXd x1, VectorXd x2) {
            VectorXd g = VectorXd::Zero(pars_.rows());
            return g;
        };

    protected:
        VectorXd pars_;

    };

    class IsotropicGaussianKernel : public Kernel {

    public:

        IsotropicGaussianKernel() {};
        IsotropicGaussianKernel(VectorXd pars) : Kernel (pars) {};

        virtual double evaluate (VectorXd x1, VectorXd x2) {
            int jp1 = int(0.5 * x1.rows()) + 2;
            VectorXd d = x1 - x2;
            double chi2 = d.dot(d),
                   r = sqrt(chi2)/fabs(pars_[2]),
                   omr = 1.0 - r, f;

            // Compute the compact envelope function.
            if (r >= 1.0) return 0.0;
            f = pow(omr, jp1)*(jp1*r + 1);

            // Compute the squared-exp covariance.
            chi2 /= pars_[1] * pars_[1];
            return pars_[0] * pars_[0] * exp(-0.5 * chi2) * f;
        };

        virtual VectorXd gradient (VectorXd x1, VectorXd x2) {
            int jp1 = int(0.5 * x1.rows()) + 2;
            VectorXd d = x1 - x2,
                     grad(pars_.rows());
            double e, value,
                   norm = d.dot(d),
                   r = sqrt(norm)/fabs(pars_[2]),
                   omr = 1.0 - r, f, fv;

            // Compute the compact kernel.
            if (r >= 1.0) return VectorXd::Zero(pars_.rows());
            f = pow(omr, jp1)*(jp1*r + 1);

            // Compute the gradient.
            e = -0.5 * norm / pars_[1] / pars_[1];
            value = exp(e);
            fv = value * f;
            grad(0) = 2 * pars_[0] * fv;
            grad(1) = -2 * e * pars_[0] * fv;
            grad(2) = pars_[0] * pars_[0] * value * r / fabs(pars_[2]);
            return grad;
        };

    };

    template <class KernelType>
    class GaussianProcess {

    private:
        KernelType kernel_;
        int info_;
        bool computed_;
        MatrixXd x_;
        SimplicialLDLT<SparseMatrix<double> > *L_;

    public:
        GaussianProcess () {
            info_ = 0;
            computed_ = false;
        };

        KernelType kernel () const { return kernel_; };
        void set_kernel (KernelType k) {
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
                                    kernel_.evaluate(x.row(i), x.row(i))
                                    + yerr[i]*yerr[i]));
                for (j = i + 1; j < nsamples; ++j) {
                    value = kernel_.evaluate(x.row(i), x.row(j));
                    if (value > 0) entries.push_back(T(j, i, value));
                }
            }
            Kxx.setFromTriplets(entries.begin(), entries.end());
            Kxx.makeCompressed();

            // Factorize the covariance.
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
            int i, j, k, nsamples = y.rows(), npars = kernel_.npars();
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
                    grad = kernel_.gradient(x_.row(i), x_.row(j));
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
                    Kxs(i, j) = kernel_.evaluate(x_.row(i), x.row(j));

            (*cov).resize(ntest, ntest);
            for (i = 0; i < ntest; ++i) {
                (*cov)(i, i) = kernel_.evaluate(x.row(i), x.row(i));
                for (j = i + 1; j < ntest; ++j) {
                    value = kernel_.evaluate(x.row(i), x.row(j));
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
