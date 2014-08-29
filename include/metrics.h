#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>

using std::vector;

namespace george {
namespace metrics {

class IsotropicMetric {
public:

    IsotropicMetric (const unsigned int ndim, const double scale)
        : ndim_(ndim), vector_(1)
    {
        vector_[0] = scale;
    };

    double get_squared_distance (const double* x1, const double* x2) const {
        int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            r2 += d*d;
        }
        return r2 / vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        double r2 = get_squared_distance(x1, x2);
        grad[0] = -r2 / vector_[0];
        return r2;
    };

    // Parameter vector spec.
    unsigned int size () const { return 1; };
    void set_parameter (const unsigned int i, const double value) {
        vector_[i] = value;
    };
    double get_parameter (const unsigned int i) const {
        return vector_[i];
    };

private:
    unsigned int ndim_;
    vector<double> vector_;

};

class AxisAlignedMetric {
public:

    AxisAlignedMetric (const unsigned int ndim, const double* scales)
        : ndim_(ndim), vector_(ndim)
    {
        unsigned int i;
        for (i = 0; i < ndim; ++i) set_parameter(i, scales[0]);
    };

    double get_squared_distance (const double* x1, const double* x2) const {
        int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            r2 += d * d / vector_[i];
        }
        return r2;
    };

    double gradient (const double* x1, const double* x2, double* grad) const {
        int i;
        double d, r2 = 0.0;
        for (i = 0; i < ndim_; ++i) {
            d = x1[i] - x2[i];
            d = d * d / vector_[i];
            r2 += d;
            grad[i] = -d / vector_[i];
        }
        return r2;
    };

    // Parameter vector spec.
    unsigned int size () const { return ndim_; };
    void set_parameter (const unsigned int i, const double value) {
        vector_[i] = value;
    };
    double get_parameter (const unsigned int i) const {
        return vector_[i];
    };

private:
    unsigned int ndim_;
    double inverse_scale_;
    vector<double> vector_;

};

}; // namespace metrics
}; // namespace george

#endif
