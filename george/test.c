#include "time.h"
#include "george.h"
#include "math.h"


int main ()
{
    clock_t begin, end;
    double time_spent;

    int v, version[3];

#ifdef CHOLMOD_HAS_VERSION_FUNCTION
    cholmod_version (version);
    printf("CHOLMOD version: %d.%d.%d\n", version[0], version[1], version[2]);
#else
    v = CHOLMOD_VERSION;
    printf("CHOLMOD version: %d\n", v);
#endif

    // Generate time series.
    int i, j, ndata = 5000;
    double t,
           *x = malloc(ndata * sizeof(double)),
           *y = malloc(ndata * sizeof(double)),
           *yerr = malloc(ndata * sizeof(double));
    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 60.) {
        x[i] = t + 0.01 * sin(t);
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

    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 60.)
        y[i] = 5 * sin(t);

    begin = clock();
    lnlike = george_log_likelihood (y, gp);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("lnlike: %f (%f seconds)\n", lnlike, time_spent);

    free (x);
    free (yerr);
    george_free_gp(gp);
    return 0;
}
