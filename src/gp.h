#ifndef _GP_H
#define _GP_H

#include <Eigen/Dense>

int evaluateGP(Eigen::MatrixXd x,
               Eigen::VectorXd y,
               Eigen::VectorXd sigma,
               Eigen::MatrixXd target,

               Eigen::VectorXd pars,
               double (*kernel) (Eigen::VectorXd, Eigen::VectorXd,
                                 Eigen::VectorXd),

               Eigen::VectorXd *mean,
               Eigen::MatrixXd *cov,
               double *loglike);

//
// Kernels.
//


double isotropicKernel (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);
double diagonalKernel  (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);

#endif;
