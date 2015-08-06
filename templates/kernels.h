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
    virtual unsigned size () const { return 0; }
    virtual unsigned get_ndim () const { return 0; }
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
    unsigned size () const { return kernel1_->size() + kernel2_->size(); };
    unsigned get_ndim () const { return kernel1_->get_ndim(); }
    void set_parameter (const unsigned i, const double value) {
        unsigned n = kernel1_->size();
        if (i < n) kernel1_->set_parameter(i, value);
        else kernel2_->set_parameter(i-n, value);
    };
    double get_parameter (const unsigned i) const {
        unsigned n = kernel1_->size();
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
        unsigned n = this->kernel1_->size();
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
        unsigned i, n1 = this->kernel1_->size(), n2 = this->size();
        this->kernel1_->gradient(x1, x2, grad);
        this->kernel2_->gradient(x1, x2, &(grad[n1]));
        double k1 = this->kernel1_->value(x1, x2),
               k2 = this->kernel2_->value(x1, x2);
        for (i = 0; i < n1; ++i) grad[i] *= k2;
        for (i = n1; i < n2; ++i) grad[i] *= k1;
    };
};

{% for spec in specs %}
/*
{{ spec.doc -}}
*/
{%- if spec.stationary %}

class {{ spec.name }} : public Kernel {
public:
    {{ spec.name }} (
        {%- for param in spec.params %}
        double {{ param }},
        {%- endfor %}
        Metric* metric
    ) :
        size_({{ spec.params|length }}),
        metric_(metric)
        {% for param in spec.params -%}
        , param_{{ param }}_({{param}})
        {% endfor -%} {};
    ~{{ spec.name }} () { delete metric_; };

    unsigned get_ndim () const { return this->metric_->get_ndim(); };

    double get_parameter (unsigned i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return this->param_{{ param }}_;
        {% endfor -%}
        return this->metric_->get_parameter(i - this->size_);
    };
    void set_parameter (unsigned i, double value) {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) this->param_{{ param }}_ = value; else
        {% endfor -%}
        this->metric_->set_parameter(i - this->size_, value);
    };

    template <typename T>
    T get_value (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            T r2) {
        {{ spec.value | indent(8) }}
    };

    double value (const double* x1, const double* x2) {
        double r2 = this->metric_->value(x1, x2);
        return this->get_value(
            {% for param in spec.params -%}
            this->param_{{ param }}_,
            {% endfor -%}
            r2);
    };

    {% for param in spec.params -%}
    double {{ param }}_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            double r2) {
        {{ spec.grad[param] | indent(8) }}
    };
    {% endfor -%}

    void gradient (const double* x1, const double* x2, double* grad) {
        unsigned i, n = this->size();
        double r2 = this->metric_->value(x1, x2);
        Jet<double> value = this->get_value(
                {% for param in spec.params -%}
                this->param_{{ param }}_,
                {% endfor %}
                Jet<double>(r2, 1.0));

        {% for param in spec.params -%}
        grad[{{ loop.index - 1 }}] = {{ param }}_gradient(
                {% for param in spec.params -%}
                this->param_{{ param }}_,
                {% endfor -%}
                r2);
        {% endfor %}
        this->metric_->gradient(x1, x2, &(grad[this->size_]));
        for (i = this->size_; i < n; ++i) grad[i] *= value.v;
    };

    unsigned size () const { return this->metric_->size() + this->size_; };

private:
    unsigned size_;
    Metric* metric_;
    {% for param in spec.params -%}
    double param_{{ param }}_;
    {% endfor %}
};

{% else %}

class {{ spec.name }} : public Kernel {
public:
    {{ spec.name }} (
        {%- for param in spec.params %}
        double {{ param }},
        {%- endfor %}
        Subspace* subspace
    ) :
        size_({{ spec.params|length }}),
        subspace_(subspace)
        {%- for param in spec.params %}
        , param_{{ param }}_({{param}})
        {%- endfor %}
        {};
    ~{{ spec.name }} () { delete subspace_; };

    unsigned get_ndim () const { return this->subspace_->get_ndim(); };

    double get_parameter (unsigned i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return this->param_{{ param }}_;
        {% endfor -%}
        return 0.0;
    };
    void set_parameter (unsigned i, double value) {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) this->param_{{ param }}_ = value; else
        {% endfor -%};
    };

    double get_value (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            const double* x1, const double* x2) {
        {{ spec.value | indent(8) }}
    };

    double value (const double* x1, const double* x2) {
        return this->get_value(
            {% for param in spec.params -%}
            this->param_{{ param }}_,
            {% endfor -%}
            x1, x2);
    };

    {% for param in spec.params -%}
    double {{ param }}_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            const double* x1, const double* x2) {
        {{ spec.grad[param] | indent(8) }}
    };
    {% endfor -%}

    void gradient (const double* x1, const double* x2, double* grad) {
        {% for param in spec.params -%}
        grad[{{ loop.index - 1 }}] = {{ param }}_gradient(
                {% for param in spec.params -%}
                this->param_{{ param }}_,
                {% endfor -%}
                x1, x2);
        {% endfor %}
    };

    unsigned size () const { return this->size_; };

private:
    unsigned size_;
    Subspace* subspace_;
    {% for param in spec.params -%}
    double param_{{ param }}_;
    {% endfor %}
};

{% endif -%}
{% endfor -%}

}; // namespace kernels
}; // namespace george

#endif
