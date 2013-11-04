#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "george.h"

#define NBENCH 500

double benchmark (int ndata)
{
    double pars[] = {1e-3, 3.0, 8.0};

    // Generate time series.
    int i;
    double t,
           *x = malloc(ndata * sizeof(double)),
           *y = malloc(ndata * sizeof(double)),
           *yerr = malloc(ndata * sizeof(double));
    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 24.) {
        x[i] = t + 0.01 * sin(t);
        y[i] = sin(t);
        yerr[i] = 0.1;
    }

    george_gp *gp = george_allocate_gp (3, pars, NULL, *george_kernel);

    clock_t ticks = clock();
    for (i = 0; i < NBENCH; ++i) {
        george_compute (ndata, x, yerr, gp);
        george_log_likelihood (y, gp);
    }
    ticks = clock() - ticks;

    free(x);
    free(y);
    free(yerr);
    george_free_gp (gp);

    return ((double)ticks / NBENCH / CLOCKS_PER_SEC);
}

int main ()
{
    int i, N[] = {50, 100, 200, 500, 1000, 2000};
    for (i = 0; i < 6; ++i)
        printf("%d %e\n", N[i], benchmark(N[i]));

    return 0;
}
