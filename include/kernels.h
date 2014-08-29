#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <vector>

using std::vector;

namespace george {
namespace kernels {


class Kernel {
public:
    Kernel (const unsigned int ndim) : ndim_(ndim) {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) const {
        return 0.0;
    };
    virtual void gradient (const double* x1, const double* x2, double* grad) const {
        int i;
        for (i = 0; i < this->size(); ++i) grad[i] = 0.0;
    };

    // Input dimension.
    void set_ndim (const unsigned int ndim) { ndim_ = ndim; };
    unsigned int get_ndim () const { return ndim_; };

    // Parameter vector spec.
    virtual unsigned int size () const { return 0; }
    virtual void set_vector (const double* vector) {
        int i, n = this->size();
        for (i = 0; i < n; ++i) this->set_parameter(i, vector[i]);
    };
    virtual void set_parameter (const unsigned int i, const double value) {};
    virtual double get_parameter (const unsigned int i) const { return 0.0; };

protected:
    unsigned int ndim_;
};


//
// OPERATORS
//

class Operator : public Kernel {
public:
    Operator (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : kernel1_(k1), kernel2_(k2), Kernel(ndim) {};
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
    Sum (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : Operator(ndim, k1, k2) {};

    double value (const double* x1, const double* x2) const {
        return this->kernel1_->value(x1, x2) + this->kernel2_->value(x1, x2);
    };

    void gradient (const double* x1, const double* x2, double* grad) const {
        unsigned int n = this->kernel1_->size();
        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n]));
    };
};

class Product : public Operator {
public:
    Product (const unsigned int ndim, Kernel* k1, Kernel* k2)
        : Operator(ndim, k1, k2) {};

    double value (const double* x1, const double* x2) const {
        return this->kernel1_->value(x1, x2) * this->kernel2_->value(x1, x2);
    };

    void gradient (const double* x1, const double* x2, double* grad) const {
        unsigned int i, n1 = this->kernel1_->size(), n2 = this->size();

        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n1]));

        double k1 = this->kernel1_->value(x1, x2),
               k2 = this->kernel2_->value(x1, x2);
        for (i = 0; i < n1; ++i) grad[i] *= k2;
        for (i = n1; i < n2; ++i) grad[i] *= k1;
    };
};


//
// RADIAL KERNELS
//

template <typename M>
class RadialKernel : public Kernel {
public:
    RadialKernel (const long ndim, M* metric)
        : Kernel(ndim), metric_(metric) {};
    ~RadialKernel () {
        delete metric_;
    };

    // Interface to the metric.
    double get_squared_distance (const double* x1, const double* x2) const {
        return metric_->get_squared_distance(x1, x2);
    };
    virtual double get_radial_gradient (double r2) const {
        return 0.0;
    };

    virtual void gradient (const double* x1, const double* x2, double* grad) const {
        int i, n = metric_->size();
        double r2 = metric_->gradient(x1, x2, grad),
               kg = this->get_radial_gradient(r2);
        for (i = 0; i < n; ++i) grad[i] *= kg;
    };

    // Parameter vector spec.
    unsigned int size () const { return metric_->size(); };
    void set_parameter (const unsigned int i, const double value) {
        metric_->set_parameter(i, value);
    };
    double get_parameter (const unsigned int i) const {
        return metric_->get_parameter(i);
    };

protected:
    M* metric_;

};

template <typename M>
class ExpSquaredKernel : public RadialKernel<M> {
public:
    ExpSquaredKernel (const long ndim, M* metric)
        : RadialKernel<M>(ndim, metric) {};
    double value (const double* x1, const double* x2) const {
        return exp(-0.5 * this->get_squared_distance(x1, x2));
    };
    double get_radial_gradient (double r2) const {
        return -0.5 * exp(-0.5 * r2);
    };

};

template <typename M>
class ExpKernel : public RadialKernel<M> {
public:
    ExpKernel (const long ndim, M* metric) : RadialKernel<M>(ndim, metric) {};
    double value (const double* x1, const double* x2) const {
        return exp(-sqrt(this->get_squared_distance(x1, x2)));
    };
    double get_radial_gradient (double r2) const {
        double r = sqrt(r2);
        return -0.5 * exp(-r) / r;
    };
};

}; // namespace kernels
}; // namespace george

#endif
