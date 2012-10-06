#ifndef _GP_H
#define _GP_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

int evaluateGP(Eigen::MatrixXd x,
               Eigen::MatrixXd y,
               Eigen::VectorXd sigma,
               Eigen::MatrixXd target,
               Eigen::VectorXd pars,
               double (*kernel) (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd),
               Eigen::VectorXd *mean,
               Eigen::VectorXd *variance,
               double *loglike,
               double sparsetol);

//
// Kernels.
//


double isotropicKernel (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);
double diagonalKernel  (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);

#endif;
