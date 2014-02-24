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

    SparseKernel (double alpha, double scale, double full_width) {
        set_alpha (alpha);
        set_scale (scale);
        set_full_width (full_width);
    };
    SparseKernel (const double* params) { set_params (params); };

    //
    // Setters for the hyperparameters.
    //
    void set_alpha (double v) { a2_ = v*v; a_ = sqrt(a2_); };
    void set_scale (double v) { s2_ = v*v; s_ = sqrt(s2_); };
    void set_full_width (double v) { fw2_ = v*v; fw_ = sqrt(fw2_); };
    void set_params (const double* p) {
        set_alpha (p[0]);
        set_scale (p[1]);
        set_full_width (p[2]);
    };

    //
    // Evaluate the kernel and optionally the gradient.
    //
    double evaluate (double x1, double x2, int *flag) const {
        double d = x1 - x2, chi2 = d * d, r, omr, k0, k;

        // If the distance is greater than the support, bail.
        *flag = 0;
        if (chi2 >= fw2_) return 0.0;

        // Compute the kernel value.
        *flag = 1;
        r = sqrt(chi2 / fw2_);
        omr = 1.0 - r;
        k0 = a2_ * exp(-0.5 * chi2 / s2_);
        k = k0 * omr * omr * (2*r + 1);

        return k;
    };

private:

    double a_, a2_, s_, s2_, fw_, fw2_;

};

};

#endif
