#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "george.h"

#define EPS (1e-6)
#define TOL (1e-5)


int main ()
{
    double pars[] = {1e-3, 3.0, 8.0};

    // Generate time series.
    int i, j, ndata = 500, count = 0;
    double t,
           *x = malloc(ndata * sizeof(double)),
           *y = malloc(ndata * sizeof(double)),
           *yerr = malloc(ndata * sizeof(double));
    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 60.) {
        x[i] = t + 0.01 * sin(t);
        y[i] = sin(t);
        yerr[i] = 0.1;
    }

    george_gp *gp = george_allocate_gp (3, pars, NULL, *george_kernel);

    int info = george_compute (ndata, x, yerr, gp);
    double ll0 = george_log_likelihood (y, gp), llp, llm, diff,
           grad[3];

    george_grad_log_likelihood (y, grad, gp);

    for (i = 0; i < 3; ++i) {
        pars[i] += EPS;
        george_compute (ndata, x, yerr, gp);
        llp = george_log_likelihood (y, gp);
        pars[i] -= 2 * EPS;
        george_compute (ndata, x, yerr, gp);
        llm = george_log_likelihood (y, gp);
        pars[i] += EPS;

        diff = fabs((grad[i] - 0.5 * (llp - llm) / EPS) / grad[i]);

        printf("    Parameter %d - analytical: %15.8e    " \
               "numerical: %15.8e    rel diff: %15.8e\t",
               i+1, grad[i], 0.5 * (llp - llm) / EPS, diff);

        if (diff < TOL) {
            printf("[passed]\n");
            count += 1;
        } else {
            printf("[failed]\n");
            return -1;
        }
    }

    return 0;
}
