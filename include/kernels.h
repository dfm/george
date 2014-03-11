#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>

namespace george {
namespace kernels {

//
// Abstract kernel base class.
//
class Kernel {
public:
    virtual ~Kernel () {};

    virtual double evaluate (const double& x1, const double& x2, int *flag) const {
        *flag = 0;
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

    double evaluate (const double& x1, const double& x2, int *flag) const {
        int f1, f2;
        double k1 = kernel1_->evaluate(x1, x2, &f1),
          k2 = kernel2_->evaluate(x1, x2, &f2);
        *flag = f1 | f2;
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

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double k = kernel1_->evaluate(x1, x2, flag);
        if (!flag) return 0.0;
        return k * kernel2_->evaluate(x1, x2, flag);
    };

private:
    Kernel* kernel1_, * kernel2_;
};


//
// Standard kernels from Rasmussen & Williams Chapter 4.
//
class ConstantKernel : public Kernel {
public:
    ConstantKernel (const double* value) : value_(value[0]) {};

    double evaluate (const double& x1, const double& x2, int *flag) const {
        *flag = 1;
        return value_*value_;
    };

private:
    double value_;
};

class DotProductKernel : public Kernel {
public:
    DotProductKernel (const double* noop) {};

    double evaluate (const double& x1, const double& x2, int *flag) const {
        *flag = 1;
        return x1 * x2;
    };
};

class ExpKernel : public Kernel {
public:
    ExpKernel (const double* scale) { set_scale (scale[0]); };
    void set_scale (const double v) { s_ = fabs(v); };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double d = x1 - x2;
        *flag = 1;
        return exp(-fabs(d) / s_);
    };

private:
    double s_;
};

class ExpSquaredKernel : public Kernel {
public:
    ExpSquaredKernel (const double* scale) { set_scale (scale[0]); };
    void set_scale (const double v) { s2_ = v*v; };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double d = x1 - x2;
        *flag = 1;
        return exp(-0.5 * d * d / s2_);
    };

private:
    double s2_;
};

class CosineKernel : public Kernel {
public:
    CosineKernel (const double* period) { set_period (period[0]); };
    void set_period (const double v) { omega_ = 2 * M_PI / fabs(v); };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double d = x1 - x2;
        *flag = 1;
        return cos(omega_ * d);
    };

private:
    double omega_;
};

class ExpSine2Kernel : public Kernel {
public:
    ExpSine2Kernel (const double* params) {
        set_gamma (params[0]);
        set_period (params[1]);
    };
    void set_gamma (const double v) { gamma_ = fabs(v); };
    void set_period (const double v) { omega_ = M_PI / fabs(v); };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double d = x1 - x2, s = sin(omega_ * d);
        *flag = 1;
        return exp(-gamma_*s*s);
    };

private:
    double gamma_, omega_;
};


class Matern32Kernel : public Kernel {
public:
    Matern32Kernel (const double* scale) { set_scale (scale[0]); };
    void set_scale (const double v) { scale_ = fabs(v); };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double r = sqrt(3.0) / scale_ * fabs(x1 - x2);
        *flag = 1;
        return (1.0 + r) * exp(-r);
    };

private:
    double scale_;
};

class Matern52Kernel : public Kernel {
public:
    Matern52Kernel (const double* scale) { set_scale (scale[0]); };
    void set_scale (const double v) { scale_ = fabs(v); };

    double evaluate (const double& x1, const double& x2, int *flag) const {
        double r = sqrt(5.0) / scale_ * fabs(x1 - x2);
        *flag = 1;
        return (1.0 + r + r*r / 3.0) * exp(-r);
    };

private:
    double scale_;
};

}; // kernels
}; // george

#endif
