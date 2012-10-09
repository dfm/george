#ifndef GEORGE_GP_H
#define GEORGE_GP_H

#include <Eigen/Dense>

class GaussianProcess {

    private:

        double l2pi_;

        Eigen::VectorXd pars_;

        int ndim_, nsamples_;
        Eigen::MatrixXd x_;
        Eigen::VectorXd y_, yerr_;

        Eigen::MatrixXd Kxx_;
        Eigen::LDLT<Eigen::MatrixXd> L_;
        Eigen::VectorXd alpha_;

        double (*kernel_) (Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd);

    public:

        GaussianProcess(Eigen::VectorXd pars, double (*kernel) (
                    Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)) {
            pars_ = pars;
            kernel_ = kernel;
            l2pi_ = log(2 * M_PI);
        };

        Eigen::MatrixXd K(Eigen::MatrixXd x1, Eigen::MatrixXd x2);

        int fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd yerr);
        double evaluate();
        int predict(Eigen::MatrixXd x, Eigen::VectorXd *mean,
                                       Eigen::MatrixXd *cov);

};


#endif
// </GEORGE_GP_H>
