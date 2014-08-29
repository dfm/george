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

    // Parameter vector spec.
    unsigned int size () const { return 1; };
    const double* get_vector () const {
        return &(vector_[0]);
    };
    void set_vector (const double* vector) {
        vector_[0] = vector[0];
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
        set_vector(scales);
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

    // Parameter vector spec.
    unsigned int size () const { return ndim_; };
    const double* get_vector () const {
        return &(vector_[0]);
    };
    void set_vector (const double* vector) {
        int i;
        for (i = 0; i < ndim_; ++i) vector_[i] = vector[i];
    };

private:
    unsigned int ndim_;
    double inverse_scale_;
    vector<double> vector_;

};

}; // namespace metrics
}; // namespace george

#endif
