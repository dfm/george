#ifndef _GEORGE_METRICS_H_
#define _GEORGE_METRICS_H_

#include <cmath>
#include <vector>
#include "george/subspace.h"

#include <iostream>

namespace george {
  namespace metrics {

    //
    // This is an abstract metric base class. The subclasses have all the good
    // stuff.
    //
    class Metric {
      public:
        Metric (size_t ndim, size_t naxes, size_t size)
          : updated_(true)
          , vector_(size)
          , subspace_(ndim, naxes) {};
        virtual ~Metric () {};

        // Return the distance between two vectors.
        virtual double value (const double* x1, const double* x2) {
          return 0.0;
        };

        // Return the gradient of `value` with respect to the parameter vector.
        virtual double gradient (const double* x1, const double* x2, double* grad) {
          return 0.0;
        };

        virtual void x1_gradient (const double* x1, const double* x2, double* grad) {};

        virtual void x2_gradient (const double* x1, const double* x2, double* grad) {
          this->x1_gradient(x1, x2, grad);
          //for (size_t i = 0; i < this->subspace_.get_ndim(); ++i) {
          //  grad[i] *= -1.0;
          //}
        };

        // Parameter vector specification.
        size_t size () const { return this->vector_.size(); };
        virtual void set_parameter (size_t i, double value) {
          this->updated_ = true;
          this->vector_[i] = exp(-value);
        };
        virtual double get_parameter (size_t i) const {
          return -log(this->vector_[i]);
        };

        // Axes specification.
        void set_axis (size_t i, size_t value) {
          this->subspace_.set_axis(i, value);
        };
        size_t get_axis (size_t i) const {
          return this->subspace_.get_axis(i);
        };
        size_t get_ndim () const {
          return this->subspace_.get_ndim();
        };

      protected:
        bool updated_;
        std::vector<double> vector_;
        george::subspace::Subspace subspace_;
    };

    class IsotropicMetric : public Metric {
      public:

        IsotropicMetric (size_t ndim, size_t naxes)
          : Metric(ndim, naxes, 1) {};
        double value (const double* x1, const double* x2) {
          size_t i, j;
          double d, r2 = 0.0;
          for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            r2 += d*d;
          }
          return r2 * this->vector_[0];
        };

        double gradient (const double* x1, const double* x2, double* grad) {
          double r2 = this->value(x1, x2);
          grad[0] = -r2;
          return r2;
        };

        void x1_gradient (const double* x1, const double* x2, double* grad) {
          size_t i, j;
          for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            grad[j] = this->vector_[0] * (x1[j] - x2[j]);
          }
        };
    };

    class AxisAlignedMetric : public Metric {
      public:

        AxisAlignedMetric (size_t ndim, size_t naxes)
          : Metric(ndim, naxes, naxes) {};

        double value (const double* x1, const double* x2) {
          size_t i, j;
          double d, r2 = 0.0;
          for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            r2 += d * d * this->vector_[i];
          }
          return r2;
        };

        double gradient (const double* x1, const double* x2, double* grad) {
          size_t i, j;
          double d, r2 = 0.0;
          for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            d = x1[j] - x2[j];
            d = d * d * this->vector_[i];
            r2 += d;
            grad[i] = -d;
          }
          return r2;
        };

        void x1_gradient (const double* x1, const double* x2, double* grad) {
          size_t i, j;
          for (i = 0; i < this->subspace_.get_naxes(); ++i) {
            j = this->subspace_.get_axis(i);
            grad[j] = this->vector_[i] * (x1[j] - x2[j]);
          }
        };
    };

    //
    // Warning: Herein lie custom Cholesky functions. Use at your own risk!
    //
    inline void _custom_forward_sub (size_t n, double* L, double* b) {
      size_t i, j, k;
      for (i = 0, k = 0; i < n; ++i) {
        for (j = 0; j < i; ++j, ++k)
          b[i] -= L[k] * b[j];
        b[i] *= L[k++];  // The inverse has already been taken along the diagonal.
      }
    }

    inline void _custom_backward_sub (size_t n, double* L, double* b) {
      long long i, j;
      size_t k, k0 = (n + 1) * n / 2;
      for (i = n - 1; i >= 0; --i) {
        k = k0 - n + i;
        for (j = n-1; j > i; --j) {
          b[i] -= L[k] * b[j];
          k -= j;
        }
        b[i] *= L[k];  // The inverse has already been taken along the diagonal.
      }
    }

    class GeneralMetric : public Metric {
      public:
        GeneralMetric (size_t ndim, size_t naxes)
          : Metric(ndim, naxes, naxes*(naxes+1)/2) {};

        void set_parameter (size_t i, double value) {
          size_t j, d;
          this->updated_ = true;
          for (j = 0, d = 2; j <= i; j += d, ++d) {
            if (i == j) {
              this->vector_[i] = exp(-value);
              return;
            }
          }
          this->vector_[i] = value;
        };
        double get_parameter (size_t i) const {
          size_t j, d;
          for (j = 0, d = 2; j <= i; j += d, ++d)
            if (i == j)
              return -log(this->vector_[i]);
          return this->vector_[i];
        };

        double value (const double* x1, const double* x2) {
          size_t i, j, n = this->subspace_.get_naxes();
          double r2;
          std::vector<double> r(n);
          for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            r[i] = x1[j] - x2[j];
          }

          _custom_forward_sub(n, &(this->vector_[0]), &(r[0]));

          r2 = 0.0;
          for (i = 0; i < n; ++i) r2 += r[i] * r[i];
          return r2;
        };

        // For a matrix A, dA^-1/dt = -A^-1 dA/dt A^-1. In this case, we want
        // d(r A^-1 r)/dA = -(A^-1 r)^T (A^-1 r).
        double gradient (const double* x1, const double* x2, double* grad) {
          size_t i, j, k, n = this->subspace_.get_naxes();
          double r2;
          std::vector<double> r(n), Lir(n);
          for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            r[i] = x1[j] - x2[j];
          }

          // Compute L^{-1} r and save it.
          _custom_forward_sub(n, &(this->vector_[0]), &(r[0]));
          for (i = 0; i < n; ++i) Lir[i] = r[i];

          // Compute K^{-1} r.
          _custom_backward_sub(n, &(this->vector_[0]), &(r[0]));

          // Compute the gradient.
          for (i = 0, k = 0; i < n; ++i) {
            grad[k] = -2 * r[i] * Lir[i] * exp(this->vector_[k]);
            k++;
            for (j = i+1; j < n; ++j)
              grad[k++] = -2 * r[j] * Lir[i];
          }

          // Compute the distance.
          r2 = 0.0;
          for (i = 0; i < n; ++i) r2 += Lir[i] * Lir[i];
          return r2;
        };

        void x1_gradient (const double* x1, const double* x2, double* grad) {
          size_t i, j, n = this->subspace_.get_naxes();
          std::vector<double> r(n);
          for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            r[i] = x1[j] - x2[j];
          }

          _custom_forward_sub(n, &(this->vector_[0]), &(r[0]));

          for (i = 0; i < n; ++i) {
            j = this->subspace_.get_axis(i);
            grad[j] = r[i];
          }
        };
    };

  }; // namespace metrics
}; // namespace george

#endif
