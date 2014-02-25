#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>

namespace george {

class Kernel {

public:

    virtual ~Kernel () {};
    virtual double evaluate (double x1, double x2, int *flag) const {
        *flag = 1;
        return 0.0;
    };

};

template <typename K1, typename K2>
class MixtureKernel : public Kernel {

public:

    MixtureKernel (K1* k1, K2* k2) : kernel1_(k1), kernel2_(k2) {};
    ~MixtureKernel () {
        delete kernel1_;
        delete kernel2_;
    };
    K1* get_kernel1 () const { return kernel1_; };
    K2* get_kernel2 () const { return kernel2_; };

    double evaluate (double x1, double x2, int *flag) const {
        double k = kernel1_->evaluate(x1, x2, flag);
        if (!flag) return 0.0;
        return k + kernel2_->evaluate(x1, x2, flag);
    };

private:

    K1* kernel1_;
    K2* kernel2_;

};


template <typename K1, typename K2>
class ProductKernel : public Kernel {

public:

    ProductKernel (K1* k1, K2* k2) : kernel1_(k1), kernel2_(k2) {};
    ~ProductKernel () {
        delete kernel1_;
        delete kernel2_;
    };
    K1* get_kernel1 () const { return kernel1_; };
    K2* get_kernel2 () const { return kernel2_; };

    double evaluate (double x1, double x2, int *flag) const {
        double k = kernel1_->evaluate(x1, x2, flag);
        if (!flag) return 0.0;
        return k * kernel2_->evaluate(x1, x2, flag);
    };

private:

    K1* kernel1_;
    K2* kernel2_;

};

class ExpSquaredKernel : public Kernel {

public:

    ExpSquaredKernel (double alpha, double scale) {
        set_alpha (alpha);
        set_scale (scale);
    };
    ExpSquaredKernel (const double* params) { set_params (params); };

    //
    // Setters for the hyperparameters.
    //
    void set_alpha (double v) { a2_ = v*v; a_ = sqrt(a2_); };
    void set_scale (double v) { s2_ = v*v; s_ = sqrt(s2_); };
    void set_params (const double* p) {
        set_alpha (p[0]);
        set_scale (p[1]);
    };

    //
    // Evaluate the kernel and optionally the gradient.
    //
    double evaluate (double x1, double x2, int *flag) const {
        double d = x1 - x2, k;
        k = a2_ * exp(-0.5 * d * d / s2_);
        *flag = 1;
        return k;
    };

private:

    double a_, a2_, s_, s2_;

};

class SparseKernel : public Kernel {

public:

    SparseKernel (const double scale) {
        set_scale (scale);
    };
    SparseKernel (const double* params) { set_scale (params[0]); };

    //
    // Setters for the hyperparameters.
    //
    void set_scale (double v) { fw2_ = v*v; fw_ = sqrt(fw2_); };

    //
    // Evaluate the kernel and optionally the gradient.
    //
    double evaluate (double x1, double x2, int *flag) const {
        double d = x1 - x2, chi2 = d * d, r, omr;

        *flag = 0;
        if (chi2 >= fw2_) return 0.0;

        *flag = 1;
        r = sqrt(chi2 / fw2_);
        omr = 1.0 - r;
        return omr * omr * (2*r + 1);
    };

private:

    double fw_, fw2_;

};

};

#endif
