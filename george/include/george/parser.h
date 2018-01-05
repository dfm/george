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
    
    case 0: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::ConstantKernel (
          
          py::float_(kernel_spec.attr("log_constant")),
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 1: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::CosineKernel (
          
          py::float_(kernel_spec.attr("log_period")),
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 2: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::DotProductKernel (
          
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 3: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::EmptyKernel (
          
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 4: {
      
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::ExpKernel<metrics::IsotropicMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::ExpKernel<metrics::AxisAlignedMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::ExpKernel<metrics::GeneralMetric> (
          
          
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
      for (ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 5: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::ExpSine2Kernel (
          
          py::float_(kernel_spec.attr("gamma")),
          py::float_(kernel_spec.attr("log_period")),
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 6: {
      
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::ExpSquaredKernel<metrics::IsotropicMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::ExpSquaredKernel<metrics::AxisAlignedMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::ExpSquaredKernel<metrics::GeneralMetric> (
          
          
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
      for (ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 7: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::LinearKernel (
          
          py::float_(kernel_spec.attr("log_gamma2")),
          
          py::float_(kernel_spec.attr("order")),
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 8: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::LocalGaussianKernel (
          
          py::float_(kernel_spec.attr("location")),
          py::float_(kernel_spec.attr("log_width")),
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 9: {
      
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::Matern32Kernel<metrics::IsotropicMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::Matern32Kernel<metrics::AxisAlignedMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::Matern32Kernel<metrics::GeneralMetric> (
          
          
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
      for (ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 10: {
      
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::Matern52Kernel<metrics::IsotropicMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::Matern52Kernel<metrics::AxisAlignedMetric> (
          
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::Matern52Kernel<metrics::GeneralMetric> (
          
          
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
      for (ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 11: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::MyLocalGaussianKernel (
          
          py::float_(kernel_spec.attr("x0")),
          py::float_(kernel_spec.attr("log_w")),
          
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 12: {
      
      size_t ndim = py::int_(kernel_spec.attr("ndim"));
      py::list axes = py::list(kernel_spec.attr("axes"));
      kernel = new kernels::PolynomialKernel (
          
          py::float_(kernel_spec.attr("log_sigma2")),
          
          py::float_(kernel_spec.attr("order")),
          ndim,
          py::len(axes)
      );
      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    case 13: {
      
      py::object metric = kernel_spec.attr("metric");
      size_t metric_type = py::int_(metric.attr("metric_type"));
      size_t ndim = py::int_(metric.attr("ndim"));
      py::list axes = py::list(metric.attr("axes"));
      bool blocked = py::bool_(kernel_spec.attr("blocked"));
      py::array_t<double> min_block = py::array_t<double>(kernel_spec.attr("min_block"));
      py::array_t<double> max_block = py::array_t<double>(kernel_spec.attr("max_block"));

      // Select the correct template based on the metric type
      if (metric_type == 0) {
        kernel = new kernels::RationalQuadraticKernel<metrics::IsotropicMetric> (
          
          py::float_(kernel_spec.attr("log_alpha")),
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 1) {
        kernel = new kernels::RationalQuadraticKernel<metrics::AxisAlignedMetric> (
          
          py::float_(kernel_spec.attr("log_alpha")),
          
          blocked,
          (double*)&(min_block.unchecked<1>()(0)),
          (double*)&(max_block.unchecked<1>()(0)),
          ndim,
          py::len(axes)
        );
      } else if (metric_type == 2) {
        kernel = new kernels::RationalQuadraticKernel<metrics::GeneralMetric> (
          
          py::float_(kernel_spec.attr("log_alpha")),
          
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
      for (ssize_t i = 0; i < data.shape(0); ++i) {
        kernel->set_metric_parameter(i, data(i));
      }

      

      for (size_t i = 0; i < py::len(axes); ++i) {
        kernel->set_axis(i, py::int_(axes[py::int_(i)]));
      }

      break; }
    
    default:
      throw std::invalid_argument("unrecognized kernel");
  }

  return kernel;
}

}

#endif