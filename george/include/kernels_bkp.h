#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>

#include "metrics.h"
#include "subspace.h"
#include "autodiff.h"

using std::vector;
using george::autodiff::Jet;
using george::metrics::Metric;
using george::subspace::Subspace;

namespace george {

namespace kernels {

class Kernel {
public:
    Kernel () {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) { return 0.0; };
    virtual void gradient (const double* x1, const double* x2, double* grad) {};

    // Parameter vector spec.
    virtual unsigned int size () const { return 0; }
    virtual void set_parameter (unsigned i, double v) {};
    virtual double get_parameter (unsigned i) const { return 0.0; };
};


// class CustomKernel : public Kernel {
// public:
//     CustomKernel (const unsigned int ndim, const unsigned int size, void* meta,
//                   double (*f) (const double* pars, const unsigned int size,
//                                void* meta,
//                                const double* x1, const double* x2,
//                                const unsigned int ndim),
//                   void (*g) (const double* pars, const unsigned int size,
//                              void* meta,
//                              const double* x1, const double* x2,
//                              const unsigned int ndim, double* grad))
//         : ndim_(ndim), size_(size), meta_(meta), f_(f), g_(g)
//     {
//         parameters_ = new double[size];
//     };
//     ~CustomKernel () {
//         delete parameters_;
//     };
//
//     // Call the external functions.
//     double value (const double* x1, const double* x2) {
//         return f_(parameters_, size_, meta_, x1, x2, this->get_ndim());
//     };
//     void gradient (const double* x1, const double* x2, double* grad) {
//         g_(parameters_, size_, meta_, x1, x2, this->get_ndim(), grad);
//     };
//
//     // Parameters.
//     unsigned int size () const { return size_; }
//     void set_parameter (const unsigned int i, const double value) {
//         parameters_[i] = value;
//     };
//     double get_parameter (const unsigned int i) const {
//         return parameters_[i];
//     };
//
// protected:
//     double* parameters_;
//     unsigned int ndim_, size_;
//
//     // Metadata needed for this function.
//     void* meta_;
//
//     // The function and gradient pointers.
//     double (*f_) (const double* pars, const unsigned int size, void* meta,
//                   const double* x1, const double* x2,
//                   const unsigned int ndim);
//     void (*g_) (const double* pars, const unsigned int size, void* meta,
//                 const double* x1, const double* x2,
//                 const unsigned int ndim, double* grad);
// };


//
// OPERATORS
//

class Operator : public Kernel {
public:
    Operator (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~Operator () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    // Parameter vector spec.
    unsigned int size () const { return kernel1_->size() + kernel2_->size(); };
    void set_parameter (const unsigned int i, const double value) {
        unsigned int n = kernel1_->size();
        if (i < n) kernel1_->set_parameter(i, value);
        else kernel2_->set_parameter(i-n, value);
    };
    double get_parameter (const unsigned int i) const {
        unsigned int n = kernel1_->size();
        if (i < n) return kernel1_->get_parameter(i);
        return kernel2_->get_parameter(i-n);
    };

protected:
    Kernel* kernel1_, * kernel2_;
};

class Sum : public Operator {
public:
    Sum (Kernel* k1, Kernel* k2) : Operator(k1, k2) {};
    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) + this->kernel2_->value(x1, x2);
    };
    void gradient (const double* x1, const double* x2, double* grad) {
        unsigned int n = this->kernel1_->size();
        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n]));
    };
};

class Product : public Operator {
public:
    Product (Kernel* k1, Kernel* k2) : Operator(k1, k2) {};
    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) * this->kernel2_->value(x1, x2);
    };
    void gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i, n1 = this->kernel1_->size(), n2 = this->size();
        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n1]));
        double k1 = this->kernel1_->value(x1, x2),
               k2 = this->kernel2_->value(x1, x2);
        for (i = 0; i < n1; ++i) grad[i] *= k2;
        for (i = n1; i < n2; ++i) grad[i] *= k1;
    };
};

{% for spec in specs %}
{% if spec.stationary %}


class ExpSine2Kernel : public Kernel {
public:
    ExpSine2Kernel (
        double gamma,
        double period,
        Metric* metric
    ) :
        param_gamma_(gamma),
        param_period_(period),
        size_(2),
        metric_(metric) {};
    ~ExpSine2Kernel () { delete metric_; };

    double get_parameter (unsigned i) const {
        if (i == 0) return this->param_gamma_;
        if (i == 1) return this->param_period_;
        return this->metric_->get_parameter(i - this->size_);
    };
    void set_parameter (unsigned i, double value) {
        if (i == 0) this->param_gamma_ = value;
        else if (i == 1) this->param_period_ = value;
        else this->metric_->set_parameter(i - this->size_, value);
    };

    template <typename T>
    T get_value (double gamma, double period, T r2) {
        T s = sin(M_PI * sqrt(r2) / period);
        return exp(-gamma * s * s);
    }

    double value (const double* x1, const double* x2) {
        double r2 = this->metric_->value(x1, x2);
        return this->get_value(param_gamma_, param_period_, r2);
    };

    double gamma_gradient (double gamma, double period, double r2) {
        double s = sin(M_PI * sqrt(r2) / period), s2 = s * s;
        return -s2 * exp(-gamma * s2);
    };

    double period_gradient (double gamma, double period, double r2) {
        double arg = M_PI * sqrt(r2) / period, s = sin(arg), c = cos(arg),
               A = exp(-gamma * s * s);
        return 2 * gamma * arg * c * s * A / period;
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        double r2 = this->metric_->value(x1, x2);
        Jet<double> value = this->get_value(param_gamma_, param_period_,
                                            Jet<double>(r2, 1.0));
        grad[0] = gamma_gradient (param_gamma_, param_period_, r2);
        grad[1] = period_gradient (param_gamma_, param_period_, r2);
        this->metric_->gradient(x1, x2, &(grad[2]));
        unsigned i, n = this->size();
        for (i = 2; i < n; ++i) grad[i] *= value.v;
    };

    unsigned size () const { return this->metric_->size() + this->size_; };

private:
    unsigned size_;
    Metric* metric_;
    double param_gamma_;
    double param_period_;
};

{% else %}

class ConstantKernel : public Kernel {
public:
    ConstantKernel (
        double value,
        Subspace* subspace
    ) :
        param_value_(value),
        size_(1),
        subspace_(subspace) {};
    ~ConstantKernel () { delete subspace_; };

    double get_parameter (unsigned i) const {
        if (i == 0) return this->param_value_;
        return 0.0;
    };
    void set_parameter (unsigned i, double value) {
        if (i == 0) this->param_value_ = value;
    };

    double get_value (double value, const double* x1, const double* x2) {
        return value;
    }

    double value (const double* x1, const double* x2) {
        return this->get_value(param_value_, x1, x2);
    };

    double value_gradient (double value, const double* x1, const double* x2) {
        return 1.0;
    };

    void gradient (const double* x1, const double* x2, double* grad) {
        grad[0] = value_gradient (param_value_, x1, x2);
    };

    unsigned size () const { return this->size_; };

private:
    unsigned size_;
    Subspace* subspace_;
    double param_value_;
};

{% endif %}
{% endfor %}

}; // namespace kernels
}; // namespace george

#endif
