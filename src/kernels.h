#ifndef GEORGE_KERNELS_H
#define GEORGE_KERNELS_H

#include <Eigen/Dense>

double isotropicKernel (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);
double diagonalKernel  (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);

#endif
// </GEORGE_KERNELS_H>
