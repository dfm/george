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

class SumKernel : public Kernel {

public:

    SumKernel (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~SumKernel () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    double evaluate (double x1, double x2, int *flag) const {
        double k = kernel1_->evaluate(x1, x2, flag);
        if (!flag) return 0.0;
        return k + kernel2_->evaluate(x1, x2, flag);
    };

private:

    Kernel* kernel1_, * kernel2_;

};


class ProductKernel : public Kernel {

public:

    ProductKernel (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~ProductKernel () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    double evaluate (double x1, double x2, int *flag) const {
        double k = kernel1_->evaluate(x1, x2, flag);
        if (!flag) return 0.0;
        return k * kernel2_->evaluate(x1, x2, flag);
    };

private:

    Kernel* kernel1_, * kernel2_;

};

class ExpKernel : public Kernel {

public:

    ExpKernel (double alpha, double scale) {
        set_alpha (alpha);
        set_scale (scale);
    };
    ExpKernel (const double* params) { set_params (params); };

    //
    // Setters for the hyperparameters.
    //
    void set_alpha (double v) { a2_ = v*v; a_ = sqrt(a2_); };
    void set_scale (double v) { s_ = fabs(v); };
    void set_params (const double* p) {
        set_alpha (p[0]);
        set_scale (p[1]);
    };

    //
    // Evaluate the kernel and optionally the gradient.
    //
    double evaluate (double x1, double x2, int *flag) const {
        double d = x1 - x2;
        *flag = 1;
        return a2_ * exp(-fabs(d) / s_);
    };

private:

    double a_, a2_, s_;

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

class CosineKernel : public Kernel {

public:

    CosineKernel (const double period) {
        set_period (period);
    };
    CosineKernel (const double* params) { set_period (params[0]); };

    //
    // Setters for the hyperparameters.
    //
    void set_period (double v) { omega_ = 2 * M_PI / fabs(v); };

    //
    // Evaluate the kernel and optionally the gradient.
    //
    double evaluate (double x1, double x2, int *flag) const {
        double d = x1 - x2;
        *flag = 1;
        return cos(omega_ * d);
    };

private:

    double omega_;

};

};

#endif
