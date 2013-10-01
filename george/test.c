#include "time.h"


int main ()
{
    clock_t begin, end;
    double time_spent;

    // Generate time series.
    int i, j, ndata = 2000;
    double t,
           *x = malloc(ndata * sizeof(double)),
           *y = malloc(ndata * sizeof(double)),
           *yerr = malloc(ndata * sizeof(double));
    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 60.) {
        x[i] = t;
        y[i] = sin(t);
        yerr[i] = 0.1;
    }

    double pars[] = {1e-3, 3.0, 8.0};
    george_gp *gp = george_allocate_gp (pars, *george_kernel);
    printf("initialized\n");

    begin = clock();
    int info = george_compute (ndata, x, yerr, gp);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("computed: %d (%f seconds)\n", info, time_spent);

    begin = clock();
    double lnlike = george_log_likelihood (y, gp);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("lnlike: %f (%f seconds)\n", lnlike, time_spent);

    free (x);
    free (yerr);
    george_free_gp(gp);
    return 0;
}
