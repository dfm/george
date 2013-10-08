#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "george.h"

#define EPS (1e-6)
#define TOL (1e-5)

int main ()
{
    int i, j, nin = 10, nout = 20;
    double pars[] = {1.0, 4.0, 8.0};

    // Generate some fake data.
    double *x = malloc(nin * sizeof(double)),
           *y = malloc(nin * sizeof(double)),
           *yerr = malloc(nin * sizeof(double)),
           *t = malloc(nout * sizeof(double)),
           *mean = malloc(nout * sizeof(double)),
           *cov = malloc(nout*nout * sizeof(double));

    for (i = 0; i < nin; ++i) {
        x[i] = (double)i * 10.0 / nin;
        y[i] = sin(x[i]);
        yerr[i] = 0.01;
    }
    for (i = 0; i < nout; ++i) t[i] = (double)i * 10.0 / nout;

    // Set up the Gaussian process.
    george_gp *gp = george_allocate_gp (3, pars, NULL, *george_kernel);
    int info = george_compute(nin, x, yerr, gp);

    // Compute the prediction.
    info = george_predict (y, nout, t, mean, 1, cov, gp);

    for (i = 0; i < nout; ++i)
        for (j = 0; j < nout; ++j)
            printf("%d %d %f\n", i, j, cov[i*nout+j]);

    return 0;
}
