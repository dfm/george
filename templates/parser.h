#ifndef _GEORGE_PARSER_H_
#define _GEORGE_PARSER_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "george/kernels.h"
#include "george/exceptions.h"

namespace george {

namespace py = pybind11;

kernels::Kernel* parse_kernel_spec (const py::object& kernel_spec) {

  if (!py::hasattr(kernel_spec, "is_kernel")) throw std::invalid_argument("invalid kernel");

  // Deal with operators first
  bool is_kernel = py::bool_(kernel_spec.attr("is_kernel"));
  if (!is_kernel) {
    kernels::Kernel *k1, *k2;
    py::object spec1 = kernel_spec.attr("k1"),
               spec2 = kernel_spec.attr("k2");
    k1 = parse_kernel_spec(spec1);
    k2 = parse_kernel_spec(spec2);
    if (k1->get_ndim() != k2->get_ndim()) throw dimension_mismatch();
    size_t op = py::int_(kernel_spec.attr("operator_type"));
    if (op == 0) {
      return new kernels::Sum(k1, k2);
    } else if (op == 1) {
      return new kernels::Product(k1, k2);
    } else {
      throw std::invalid_argument("unrecognized operator");
    }
  }


  kernels::Kernel* kernel;
  size_t kernel_type = py::int_(kernel_spec.attr("kernel_type"));
  switch (kernel_type) {
    {% for spec in specs %}
    case {{ spec.index }}: {
      {% if spec.stationary %}
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::{{ spec.name }}<metrics::IsotropicMetric> (
          {% for param in spec.params %}
          py::float_(kernel_spec.attr("{{ param }}")),
          {%- endfor %}
          {% for con in spec.constants %}
          py::float_(kernel_spec.attr("{{ con.name }}")),
          {%- endfor %}
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::{{ spec.name }}<metrics::AxisAlignedMetric> (
          {% for param in spec.params %}
          py::float_(kernel_spec.attr("{{ param }}")),
          {%- endfor %}
          {% for con in spec.constants %}
          py::float_(kernel_spec.attr("{{ con.name }}")),
          {%- endfor %}
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::{{ spec.name }}<metrics::GeneralMetric> (
          {% for param in spec.params %}
          py::float_(kernel_spec.attr("{{ param }}")),
          {%- endfor %}
          {% for con in spec.constants %}
          py::float_(kernel_spec.attr("{{ con.name }}")),
          {%- endfor %}
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else {
        throw std::invalid_argument("unrecognized metric");
      }

      // Get the parameters
      py::function f = py::function(metric.attr("get_parameter_vector"));
      py::array_t<double> vector = py::array_t<double>(f(true));
      auto data = vector.unchecked<1>();
      for (py::ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      {% else %}
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::{{ spec.name }} (
          {% for param in spec.params %}
          py::float_(kernel_spec.attr("{{ param }}")),
          {%- endfor %}
          {% for con in spec.constants %}
          py::float_(kernel_spec.attr("{{ con.name }}")),
          {%- endfor %}
          ndim,
          py::len(axes)
      );
      {% endif %}

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    {% endfor %}
    default:
      throw std::invalid_argument("unrecognized kernel");
  }

  return kernel;
}

}

#endif
