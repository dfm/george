#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>
#include <Eigen/Dense>

using std::vector;

namespace george {
namespace metrics {

//
// This is an abstract metric base class. The subclasses have all the good
// stuff.
//
class Metric {
public:
    Metric (const unsigned int naxes, const unsigned int size)
        : updated_(true), naxes_(naxes), axes_(naxes), vector_(size) {};
    virtual ~Metric () {};

    // Return the distance between two vectors.
    double value (const double* x1, const double* x2) {
        return 0.0;
    };

    // Return the gradient of `value` with respect to the parameter vector.
    double gradient (const double* x1, const double* x2, double* grad) {
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
        this->updated_ = true;
        this->axes_[i] = value;
    };
    unsigned int get_axis (const unsigned int i) const {
        return this->axes_[i];
    };

protected:
    bool updated_;
    unsigned int naxes_;
    vector<unsigned> axes_;
    vector<double> vector_;
};

class IsotropicMetric : public Metric {
public:

    IsotropicMetric (const unsigned int naxes) : Metric(naxes, 1) {};
    double value (const double* x1, const double* x2) {
        unsigned int i, j;
        double d, r2 = 0.0;
        for (i = 0; i < naxes_; ++i) {
            j = axes_[i];
            d = x1[j] - x2[j];
            r2 += d*d;
        }
        return r2 / this->vector_[0];
    };

    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i;
        double r2 = value(x1, x2),
               v = -r2 / this->vector_[0];
        for (i = 0; i < naxes_; ++i) grad[this->axes_[i]] = v;
        return r2;
    };
};

class AxisAlignedMetric : public Metric {
public:

    AxisAlignedMetric (const unsigned int naxes) : Metric(naxes, naxes) {};

    double value (const double* x1, const double* x2) {
        unsigned int i, j;
        double d, r2 = 0.0;
        for (i = 0; i < naxes_; ++i) {
            j = axes_[i];
            d = x1[j] - x2[j];
            r2 += d * d / this->vector_[i];
        }
        return r2;
    };

    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i, j;
        double d, r2 = 0.0;
        for (i = 0; i < naxes_; ++i) {
            j = axes_[i];
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
    GeneralMetric (const unsigned int naxes) : Metric(naxes, naxes*(naxes+1)/2) {};

    bool update () {
        unsigned int i, j, k;
        if (!(this->updated_)) return true;
        Eigen::MatrixXd A(this->naxes_, this->naxes_);
        for (i = 0, k = 0; i < this->naxes_; ++i)
            for (j = i; j < this->naxes_; ++j, ++k)
                A(j, i) = this->vector_[k];
        this->factor_ = A.ldlt();
        if (this->factor_.info() != Eigen::Success) return false;
        this->updated_ = false;
        return true;
    };

    double value (const double* x1, const double* x2) {
        unsigned int i, j;
        update();
        Eigen::VectorXd r(this->naxes_);
        for (i = 0; i < naxes_; ++i) {
            j = axes_[i];
            r(i) = x1[j] - x2[j];
        }
        return r.transpose() * this->factor_.solve(r);
    };

    double gradient (const double* x1, const double* x2, double* grad) {
        unsigned int i, j, k;
        Eigen::MatrixXd g(this->naxes_, this->naxes_);
        Eigen::VectorXd r(this->naxes_),
                        Ar(this->naxes_);
        update();
        for (i = 0; i < naxes_; ++i) {
            j = axes_[i];
            r(i) = x1[j] - x2[j];
        }
        Ar = this->factor_.solve(r);
        g = Ar * Ar.transpose();
        for (i = 0, k = 0; i < this->naxes_; ++i) {
            grad[k++] = -g(i, i);
            for (j = i+1; j < this->naxes_; ++j)
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
