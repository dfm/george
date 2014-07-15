#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using Eigen::Map;
using Eigen::Stride;
using Eigen::Dynamic;
using Eigen::Unaligned;

namespace george {
namespace kernels {

//
// Abstract kernel base class.
//
class Kernel {
public:
    virtual ~Kernel () {};
    virtual double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        return 0.0;
    };
};


//
// Kernel operations.
//
class Sum : public Kernel {
public:
    Sum (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~Sum () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        double k1 = kernel1_->evaluate(x1, x2),
          k2 = kernel2_->evaluate(x1, x2);
        return k1 + k2;
    };

private:
    Kernel* kernel1_, * kernel2_;
};

class Product : public Kernel {
public:
    Product (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~Product () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        return kernel1_->evaluate(x1, x2) * kernel2_->evaluate(x1, x2);
    };

private:
    Kernel* kernel1_, * kernel2_;
};


//
// Standard kernels from Rasmussen & Williams Chapter 4.
//
class ConstantKernel : public Kernel {
public:
    ConstantKernel (const long ndim, const double* value) : value_(value[0]) {};
    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        return value_*value_;
    };

private:
    double value_;
};

class WhiteKernel : public Kernel {
public:
    WhiteKernel (const long ndim, const double* value) : value_(value[0]) {};
    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        double d = (x1 - x2).squaredNorm();
        if (d < DBL_EPSILON) return value_*value_;
        return 0.0;
    };

private:
    double value_;
};

class DotProductKernel : public Kernel {
public:
    DotProductKernel (const long ndim, const double* noop) {};
    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        return x1.dot(x2);
    };
};

// The array `cov` here must have the form `(a, b, c, ...)` when you want a
// covariance function of the form:
//
//      a b d ...
//      b c e ...
//      d e f ...
//       ...  ...
//
class CovKernel : public Kernel {
public:
    CovKernel (const long ndim, const double* cov) {
        int n = 0;
        MatrixXd m(ndim, ndim);
        for (int i = 0; i < ndim; ++i)
            for (int j = 0; j <= i; ++j)
                m(i, j) = cov[n++];
        icov_ = Eigen::LDLT<MatrixXd, Eigen::Lower> (m);
    };

    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        VectorXd d = x1 - x2;
        double r2 = d.dot(icov_.solve(d));
        return get_value(r2);
    };

    virtual double get_value (const double r2) const {
        return exp(-0.5 * r2);
    };
    Eigen::LDLT<MatrixXd, Eigen::Lower> icov_;
};

class ExpKernel : public CovKernel {
public:
    ExpKernel (const long ndim, const double* cov) : CovKernel(ndim, cov) {};
    double get_value (const double r2) const {
        return exp(-sqrt(r2));
    };
};

class RBFKernel : public CovKernel {
public:
    RBFKernel (const long ndim, const double* cov) : CovKernel(ndim, cov) {};
    double get_value (const double r2) const {
        return exp(-0.5 * r2);
    };
};

class Matern32Kernel : public CovKernel {
public:
    Matern32Kernel (const long ndim, const double* cov) : CovKernel(ndim, cov) {};
    double get_value (const double r2) const {
        double r = sqrt(3.0 * r2);
        return (1.0 + r) * exp(-r);
    };
};

class Matern52Kernel : public CovKernel {
public:
    Matern52Kernel (const long ndim, const double* cov) : CovKernel(ndim, cov) {};
    double get_value (const double r2) const {
        double r = sqrt(5.0 * r2);
        return (1.0 + r + r*r / 3.0) * exp(-r);
    };
};

class CosineKernel : public Kernel {
public:
    CosineKernel (const int ndim, const double* period) {
        omega_ = 2 * M_PI / fabs(period[0]);
    };
    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        VectorXd d = x1 - x2;
        return cos(omega_ * sqrt(d.dot(d)));
    };

private:
    double omega_;
};

class ExpSine2Kernel : public Kernel {
public:
    ExpSine2Kernel (const int ndim, const double* params) {
        gamma_ = fabs(params[0]);
        omega_ = M_PI / fabs(params[1]);
    };
    double evaluate (const VectorXd& x1, const VectorXd& x2) const {
        VectorXd d = x1 - x2;
        double s = sin(omega_ * sqrt(d.dot(d)));
        return exp(-gamma_*s*s);
    };

private:
    double gamma_, omega_;
};

class RationalQuadraticKernel : public CovKernel {
public:
    RationalQuadraticKernel (const long ndim, const double* cov)
        : CovKernel(ndim, &(cov[1]))
    {
        alpha_ = cov[0];
    };
    double get_value (const double r2) const {
        return pow(1.0 + 0.5 * r2 / alpha_, -alpha_);
    };

private:
    double alpha_;
};

}; // kernels
}; // george

#endif
