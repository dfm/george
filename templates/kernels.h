#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>

{% for spec in specs -%}
{% for incl in spec.includes -%}
#include {{ incl }}
{% endfor %}
{%- endfor %}

#include "george/metrics.h"
#include "george/subspace.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884e+00
#endif

namespace george {

namespace kernels {

class Kernel {
public:
    Kernel () {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) { return 0.0; };
    virtual void gradient (const double* x1, const double* x2,
                           const unsigned* which, double* grad) {};
    virtual void x1_gradient (const double* x1, const double* x2,
                              double* grad) {};
    virtual void x2_gradient (const double* x1, const double* x2,
                              double* grad) {};

    // Parameter vector spec.
    virtual size_t size () const { return 0; }
    virtual size_t get_ndim () const { return 0; }
    virtual void set_parameter (size_t i, double v) {};
    virtual double get_parameter (size_t i) const { return 0.0; };
    virtual void set_metric_parameter (size_t i, double v) {};
    virtual void set_axis (size_t i, size_t v) {};
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
    size_t size () const { return kernel1_->size() + kernel2_->size(); };
    size_t get_ndim () const { return kernel1_->get_ndim(); }
    void set_parameter (size_t i, double value) {
        size_t n = kernel1_->size();
        if (i < n) kernel1_->set_parameter(i, value);
        else kernel2_->set_parameter(i-n, value);
    };
    double get_parameter (size_t i) const {
        size_t n = kernel1_->size();
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
        size_t i, n1 = this->kernel1_->size(), n2 = this->size();

        bool any = false;
        for (i = 0; i < n1; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel1_->gradient(x1, x2, which, grad);

        any = false;
        for (i = n1; i < n2; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel2_->gradient(x1, x2, &(which[n1]), &(grad[n1]));
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        this->kernel1_->x1_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x1_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) grad[i] = g1[i] + g2[i];
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        this->kernel1_->x2_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x2_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) grad[i] = g1[i] + g2[i];
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
        size_t i, n1 = this->kernel1_->size(), n2 = this->size();
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

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        double k1 = this->kernel1_->value(x1, x2);
        double k2 = this->kernel2_->value(x1, x2);
        this->kernel1_->x1_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x1_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) {
            grad[i] = k2 * g1[i] + k1 * g2[i];
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        double k1 = this->kernel1_->value(x1, x2);
        double k2 = this->kernel2_->value(x1, x2);
        this->kernel1_->x2_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x2_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) {
            grad[i] = k2 * g1[i] + k1 * g2[i];
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
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
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
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return param_{{ param }}_;
        {% endfor -%}
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) {
            param_{{ param }}_ = value;
            update_reparams();
        } else
        {% endfor -%}
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
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
            size_t i, j;
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
            constant_{{ con.name }}_,
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
        size_t i, j, n = size();
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

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
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
        double r2grad = 2.0 * radial_gradient(
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
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
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
        double r2grad = 2.0 * radial_gradient(
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
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

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

private:
    size_t size_;
    M metric_;
    int blocked_;
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
        size_t ndim,
        size_t naxes
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

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        {% for param in spec.params -%}
        if (i == {{ loop.index - 1 }}) return param_{{ param }}_;
        {% endfor -%}
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
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
            double x1, double x2) {
        {{ spec.value | indent(8) }}
    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
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
            double x1, double x2) {
        {{ spec.grad[param] | indent(8) }}
    };
    {% endfor -%}

    double _x1_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            double x1, double x2) {
        {{ spec.grad["x1"] | indent(8) }}
    };

    double _x2_gradient (
            {% for param in spec.params -%}
            double {{ param }},
            {% endfor -%}
            {% for param in spec.reparams -%}
            double {{ param }},
            {% endfor -%}
            {%- for con in spec.constants %}
            {{ con.type }} {{ con.name }},
            {%- endfor %}
            double x1, double x2) {
        {{ spec.grad["x2"] | indent(8) }}
    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        {% for param in spec.params -%}
        grad[{{ loop.index - 1 }}] = 0.0;
        {% endfor %}

        {% if spec.params -%}
        size_t i, j, n = subspace_.get_naxes();
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

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
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
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
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

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
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
