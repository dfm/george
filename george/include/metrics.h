#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "subspace.h"
#include "constants.h"

using std::vector;
using george::subspace::Subspace;

namespace george {
namespace metrics {

//
// This is an abstract metric base class. The subclasses have all the good
// stuff.
//
class Metric {
public:
    Metric (const unsigned ndim, const unsigned naxes, const unsigned size)
        : status_(METRIC_OK), updated_(true), vector_(size), subspace_(ndim, naxes) {};
    virtual ~Metric () {};

    // Return the distance between two vectors.
    virtual double value (const double* x1, const double* x2) {
        return 0.0;
    };

    // Return the gradient of `value` with respect to the parameter vector.
    virtual double gradient (const double* x1, const double* x2, double* grad) {
        return 0.0;
    };

    // Parameter vector specification.
    unsigned int size () const { return this->vector_.size(); };
    void set_parameter (const unsigned int i, const double value) {
        this->updated_ = true;
        this->vector_[i] = value;
    };
    double get_parameter (const unsigned int i) const {
        return this->vector_[i];
    };

    // Axes specification.
    void set_axis (const unsigned int i, const unsigned int value) {
        this->subspace_.set_axis(i, value);
    };
    unsigned int get_axis (const unsigned int i) const {
        return this->subspace_.get_axis(i);
    };
    unsigned int get_ndim () const {
        return this->subspace_.get_ndim();
    };

protected:
    int status_;
    bool updated_;
    vector<double> vector_;
    Subspace subspace_;
};

class IsotropicMetric : public Metric {
public:

    IsotropicMetric (const unsigned ndim, const unsigned naxes)
        : Metric(ndim, naxes, 1) {};
    double value (const double* x1, const double* x2) {
        unsigned i, j;
        double d, r2 = 0.0;
        for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            r2 += d*d;
        }
        return r2 / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned i;
        double r2 = value(x1, x2),
               v = -r2 / this->vector_[0];
        for (i = 0; i < this->subspace_.get_naxes(); ++i)
            grad[this->subspace_.get_axis(i)] = v;
        return r2;
    };
};

class AxisAlignedMetric : public Metric {
public:

    AxisAlignedMetric (const unsigned ndim, const unsigned naxes)
        : Metric(ndim, naxes, naxes) {};

    double value (const double* x1, const double* x2) {
        unsigned i, j;
        double d, r2 = 0.0;
        for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            r2 += d * d / this->vector_[i];
        }
        return r2;
    };

    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned i, j;
        double d, r2 = 0.0;
        for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            d = d * d / this->vector_[i];
            r2 += d;
            grad[i] = -d / this->vector_[i];
        }
        return r2;
    };
};

class GeneralMetric : public Metric {
public:
    GeneralMetric (const unsigned ndim, const unsigned naxes)
        : Metric(ndim, naxes, naxes*(naxes+1)/2) {};

    // After the parameters have been changed, the metric matrix needs to be
    // re-factorized.
    bool update () {
        unsigned i, j, k, n = this->subspace_.get_naxes();
        if (!(this->updated_)) return true;
        Eigen::MatrixXd A(n, n);
        for (i = 0, k = 0; i < n; ++i)
            for (j = i; j < n; ++j, ++k)
                A(j, i) = this->vector_[k];
        this->factor_ = A.ldlt();
        this->updated_ = false;
        if (this->factor_.info() != Eigen::Success) return false;
        return true;
    };

    double value (const double* x1, const double* x2) {
        unsigned i, j, n = this->subspace_.get_naxes();
        update();
        Eigen::VectorXd r(n);
        for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            r(i) = x1[j] - x2[j];
        }
        return r.transpose() * this->factor_.solve(r);
    };

    // For a matrix A, dA^-1/dt = A^-1 dA/dt A^-1. In this case, we want
    // d(r A^-1 r)/dA = (A^-1 r)^T (A^-1 r). The off diagonal elements get
    // multiplied by 2 because a single parameter changes both off diagonal
    // elements.
    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned i, j, k, n = this->subspace_.get_naxes();
        Eigen::MatrixXd g(n, n);
        Eigen::VectorXd r(n), Ar(n);
        update();
        for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            r(i) = x1[j] - x2[j];
        }
        Ar = this->factor_.solve(r);
        g = Ar * Ar.transpose();
        for (i = 0, k = 0; i < n; ++i) {
            grad[k++] = -g(i, i);
            for (j = i+1; j < n; ++j)
                grad[k++] = -2*g(j, i);
        }
        return r.transpose() * Ar;
    };

private:
    Eigen::LDLT<Eigen::MatrixXd> factor_;
};

}; // namespace metrics
}; // namespace george

#endif
