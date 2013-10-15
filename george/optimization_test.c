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
    int i, j, ndata = 900, count = 0;
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

    george_optimize (ndata, x, yerr, y, 100, gp);

    for (i = 0; i < 3; ++i)
        printf("%f ", gp->pars[i]);
    printf("\n");

    return 0;
}
