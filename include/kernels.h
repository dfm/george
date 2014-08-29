#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>

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
        for (i = 0; i < size(); ++i) grad[i] = 0.0;
    };

    // Input dimension.
    void set_ndim (const unsigned int ndim) { ndim_ = ndim; };
    unsigned int get_ndim () const { return ndim_; };

    // Parameter vector spec.
    virtual unsigned int size () const { return 0; }

protected:
    unsigned int ndim_;
};


template <typename M>
class RadialKernel : public Kernel {
public:
    RadialKernel (const long ndim, M& metric)
        : Kernel(ndim), metric_(metric) {};
    double get_squared_distance (const double* x1, const double* x2) const {
        return metric_.get_squared_distance(x1, x2);
    };
    virtual double get_radial_gradient (double r2) const {
        return 0.0;
    };

    virtual void gradient (const double* x1, const double* x2, double* grad) const {
        int i, n = metric_.size();
        double r2 = metric_.gradient(x1, x2, grad),
               kg = this->get_radial_gradient(r2);
        for (i = 0; i < n; ++i) grad[i] *= kg;
    };

    // Parameter vector spec.
    unsigned int size () const { return metric_.size(); };
    const double* get_vector () const {
        return metric_.get_vector();
    };
    void set_vector (const double* vector) {
        metric_.set_vector(vector);
    };

protected:
    M& metric_;

};

template <typename M>
class ExpSquaredKernel : public RadialKernel<M> {
public:
    ExpSquaredKernel (const long ndim, M& metric)
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
    ExpKernel (const long ndim, M& metric) : RadialKernel<M>(ndim, metric) {};
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
