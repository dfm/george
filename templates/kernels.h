#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>

#include "metrics.h"
#include "subspace.h"

using std::vector;
using george::metrics::Metric;
using george::subspace::Subspace;

namespace george {

namespace kernels {

class Kernel {
public:
    Kernel () {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) { return 0.0; };
    virtual void gradient (const double* x1, const double* x2,
                           const unsigned* which, double* grad) {};

    // Parameter vector spec.
    virtual unsigned size () const { return 0; }
    virtual unsigned get_ndim () const { return 0; }
    virtual void set_parameter (unsigned i, double v) {};
    virtual double get_parameter (unsigned i) const { return 0.0; };
    virtual void set_metric_parameter (unsigned i, double v) {};
    virtual void set_axis (unsigned i, unsigned v) {};
};


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
    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        unsigned i, n1 = this->kernel1_->size(), n2 = this->size();

        bool any = false;
        for (i = 0; i < n1; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel1_->gradient(x1, x2, which, grad);

        any = false;
        for (i = n1; i < n2; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel2_->gradient(x1, x2, &(which[n1]), &(grad[n1]));
    };
};

class Product : public Operator {
public:
    Product (Kernel* k1, Kernel* k2) : Operator(k1, k2) {};
    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) * this->kernel2_->value(x1, x2);
    };
    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool any;
        unsigned i, n1 = this->kernel1_->size(), n2 = this->size();
        double k;

        any = false;
        for (i = 0; i < n1; ++i) if (which[i]) { any = true; break; }
        if (any) {
            k = this->kernel2_->value(x1, x2);
            this->kernel1_->gradient(x1, x2, which, grad);
            for (i = 0; i < n1; ++i) grad[i] *= k;
        }

        any = false;
        for (i = n1; i < n2; ++i) if (which[i]) { any = true; break; }
        if (any) {
            k = this->kernel1_->value(x1, x2);
            this->kernel2_->gradient(x1, x2, &(which[n1]), &(grad[n1]));
            for (i = n1; i < n2; ++i) grad[i] *= k;
        }
    };
};

{% for spec in specs %}
/*
{{ spec.doc -}}
*/
{%- if spec.stationary %}

template <typename M>
class {{ spec.name }} : public Kernel {
public:
    {{ spec.name }} (
        {%- for param in spec.params %}
        double {{ param }},
        {%- endfor %}
        {%- for con in spec.constants %}
        {{ con.type }} {{ con.name }},
        {%- endfor %}
        const unsigned blocked,
        const double* min_block,
        const double* max_block,
        const unsigned ndim,
        const unsigned naxes
    ) :
        size_({{ spec.params|length }}),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        {% for param in spec.params -%}
        , param_{{ param }}_({{param}})
        {% endfor -%}
        {%- for con in spec.constants %}
        , constant_{{ con.name }}_({{ con.name }})
        {%- endfor %}
    {
        unsigned i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    unsigned get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (unsigned i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return param_{{ param }}_;
        {% endfor -%}
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (unsigned i, double value) {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) {
            param_{{ param }}_ = value;
            update_reparams();
        } else
        {% endfor -%}
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (unsigned i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (unsigned i, double value) {
        metric_.set_parameter(i, value);
    };

    unsigned get_axis (unsigned i) const {
        return metric_.get_axis(i);
    };
    void set_axis (unsigned i, unsigned value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            double r2) {
        {{ spec.value | indent(8) }}
    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            unsigned i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            {% for param in spec.params -%}
            param_{{ param }}_,
            {% endfor -%}
            {% for param in spec.reparams -%}
            reparam_{{ param }}_,
            {% endfor -%}
            {%- for con in spec.constants %}
            constant_{{ con.name }}),
            {%- endfor %}
            r2);
    };

    {% for param in spec.params -%}
    double {{ param }}_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            double r2) {
        {{ spec.grad[param] | indent(8) }}
    };
    {% endfor -%}

    double radial_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            double r2) {
        {{ spec.grad["r2"] | indent(8) }}
    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        unsigned i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        {% for param in spec.params -%}
        if (which[{{ loop.index - 1 }}])
            grad[{{ loop.index - 1 }}] = {{ param }}_gradient(
                    {% for param in spec.params -%}
                    param_{{ param }}_,
                    {% endfor -%}
                    {% for param in spec.reparams -%}
                    reparam_{{ param }}_,
                    {% endfor -%}
                    {%- for con in spec.constants %}
                    constant_{{ con.name }}_,
                    {%- endfor %}
                    r2);
        {% endfor %}

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    {% for param in spec.params -%}
                    param_{{ param }}_,
                    {% endfor %}
                    {% for param in spec.reparams -%}
                    reparam_{{ param }}_,
                    {% endfor -%}
                    {%- for con in spec.constants %}
                    constant_{{ con.name }}_,
                    {%- endfor %}
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    unsigned size () const { return metric_.size() + size_; };

    void update_reparams () {
        {% for param in spec.reparams -%}
        reparam_{{ param }}_ = get_reparam_{{ param }} (
            {% for param in spec.params -%}
            param_{{ param }}_{%- if spec.constants or not loop.last %},{% endif -%}
            {% endfor %}
            {%- for con in spec.constants %}
            constant_{{ con.name }}{%- if not loop.last %},{% endif -%}
            {%- endfor %}
        );
        {% endfor %}
    };

    {% for param in spec.reparams -%}
    double get_reparam_{{ param }} (
        {% for param in spec.params -%}
        double {{ param }}{%- if spec.constants or not loop.last %},{% endif -%}
        {% endfor %}
        {%- for con in spec.constants %}
        {{ con.type }} {{ con.name }}{%- if not loop.last %},{% endif -%}
        {%- endfor %}
    ) {
        {{ spec.reparams[param] | indent(8) }}
    }
    {% endfor %}

private:
    unsigned size_;
    M metric_;
    unsigned blocked_;
    std::vector<double> min_block_, max_block_;
    {% for param in spec.params -%}
    double param_{{ param }}_;
    {% endfor %}
    {% for param in spec.reparams -%}
    double reparam_{{ param }}_;
    {% endfor %}
    {%- for con in spec.constants %}
    {{ con.type }} constant_{{ con.name }}_;
    {%- endfor %}
};

{% else %}

class {{ spec.name }} : public Kernel {
public:
    {{ spec.name }} (
        {%- for param in spec.params %}
        double {{ param }},
        {%- endfor %}
        {%- for con in spec.constants %}
        {{ con.type }} {{ con.name }},
        {%- endfor %}
        unsigned ndim,
        unsigned naxes
    ) :
        size_({{ spec.params|length }}),
        subspace_(ndim, naxes)
        {%- for param in spec.params %}
        , param_{{ param }}_({{param}})
        {%- endfor %}
        {%- for con in spec.constants %}
        , constant_{{ con.name }}_({{ con.name }})
        {%- endfor %}
    {
        update_reparams();
    };

    unsigned get_ndim () const { return subspace_.get_ndim(); };
    unsigned get_axis (const unsigned i) const { return subspace_.get_axis(i); };
    void set_axis (const unsigned i, const unsigned value) { subspace_.set_axis(i, value); };

    double get_parameter (unsigned i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return param_{{ param }}_;
        {% endfor -%}
        return 0.0;
    };
    void set_parameter (unsigned i, double value) {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) {
            param_{{ param }}_ = value;
            update_reparams();
        } else
        {% endfor -%};
    };

    double get_value (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            const double x1, const double x2) {
        {{ spec.value | indent(8) }}
    };

    double value (const double* x1, const double* x2) {
        unsigned i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                {% for param in spec.params -%}
                param_{{ param }}_,
                {% endfor -%}
                {% for param in spec.reparams -%}
                reparam_{{ param }}_,
                {% endfor -%}
                {%- for con in spec.constants %}
                constant_{{ con.name }}_,
                {%- endfor %}
                x1[j], x2[j]);
        }
        return value;
    };

    {% for param in spec.params -%}
    double {{ param }}_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            const double x1, const double x2) {
        {{ spec.grad[param] | indent(8) }}
    };
    {% endfor -%}

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        {% for param in spec.params -%}
        grad[{{ loop.index - 1 }}] = 0.0;
        {% endfor %}

        {% if spec.params -%}
        unsigned i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            {% for param in spec.params -%}
            if (which[{{ loop.index - 1}}])
                grad[{{ loop.index - 1 }}] += {{ param }}_gradient(
                    {% for param in spec.params -%}
                    param_{{ param }}_,
                    {% endfor -%}
                    {% for param in spec.reparams -%}
                    reparam_{{ param }}_,
                    {% endfor -%}
                    {%- for con in spec.constants %}
                    constant_{{ con.name }}_,
                    {%- endfor %}
                    x1[j], x2[j]);
            {% endfor %}
        }
        {% endif %}
    };

    void update_reparams () {
        {% for param in spec.reparams -%}
        reparam_{{ param }}_ = get_reparam_{{ param }} (
            {% for param in spec.params -%}
            param_{{ param }}_{%- if spec.constants or not loop.last %},{% endif -%}
            {% endfor %}
            {%- for con in spec.constants %}
            constant_{{ con.name }}_{%- if not loop.last %},{% endif -%}
            {%- endfor %}
        );
        {% endfor %}
    };

    {% for param in spec.reparams -%}
    double get_reparam_{{ param }} (
        {% for param in spec.params -%}
        double {{ param }}{%- if spec.constants or not loop.last %},{% endif -%}
        {% endfor %}
        {%- for con in spec.constants %}
        {{ con.type }} {{ con.name }}{%- if not loop.last %},{% endif -%}
        {%- endfor %}
    ) {
        {{ spec.reparams[param] | indent(8) }}
    }
    {% endfor %}

    unsigned size () const { return size_; };

private:
    unsigned size_;
    Subspace subspace_;
    {% for param in spec.params -%}
    double param_{{ param }}_;
    {% endfor %}
    {% for param in spec.reparams -%}
    double reparam_{{ param }}_;
    {% endfor %}
    {%- for con in spec.constants %}
    {{ con.type }} constant_{{ con.name }}_;
    {%- endfor %}
};

{% endif -%}
{% endfor -%}

}; // namespace kernels
}; // namespace george

#endif
