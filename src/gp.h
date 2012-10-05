#ifndef _GP_H
#define _GP_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

int evaluateGP(Eigen::VectorXd x,
               Eigen::VectorXd y,
               Eigen::VectorXd sigma,
               Eigen::VectorXd target,
               double (*kernel) (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd),
               Eigen::VectorXd *mean,
               Eigen::VectorXd *variance,
               double *loglike);

#endif;
