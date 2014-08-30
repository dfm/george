#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>

using std::vector;

namespace george {
namespace metrics {

class Metric {
public:
    Metric (const unsigned int ndim, const unsigned int size)
        : ndim_(ndim), vector_(size) {};
    virtual ~Metric () {};
    virtual double get_squared_distance (const double* x1, const double* x2) const {
        return 0.0;
    };
    virtual double gradient (const double* x1, const double* x2, double* grad) const {
        return 0.0;
    };

    // Parameter vector spec.
    virtual unsigned int size () const { return vector_.size(); };
    void set_parameter (const unsigned int i, const double value) {
        vector_[i] = value;
    };
    double get_parameter (const unsigned int i) const {
        return vector_[i];
    };

protected:
    unsigned int ndim_;
    vector<double> vector_;
};

class OneDMetric : public Metric {
public:

    OneDMetric (const unsigned int ndim, const unsigned int dim)
        : Metric(ndim, 1), dim_(dim) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        double d = x1[dim_] - x2[dim_];
        return d * d / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        double r2 = get_squared_distance(x1, x2);
        grad[0] = -r2 / this->vector_[0];
        return r2;
    };

private:
    unsigned int dim_;
};

class IsotropicMetric : public Metric {
public:

    IsotropicMetric (const unsigned int ndim) : Metric(ndim, 1) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        unsigned int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            r2 += d*d;
        }
        return r2 / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        double r2 = get_squared_distance(x1, x2);
        grad[0] = -r2 / this->vector_[0];
        return r2;
    };
};

class AxisAlignedMetric : public Metric {
public:

    AxisAlignedMetric (const unsigned int ndim) : Metric(ndim, ndim) {};

    double get_squared_distance (const double* x1, const double* x2) const {
        unsigned int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            r2 += d * d / this->vector_[i];
        }
        return r2;
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        unsigned int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            d = d * d / this->vector_[i];
            r2 += d;
            grad[i] = -d / this->vector_[i];
        }
        return r2;
    };
};

}; // namespace metrics
}; // namespace george

#endif
