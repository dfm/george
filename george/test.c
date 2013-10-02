#include "time.h"
#include "george.h"
#include "math.h"

#define EPS (1e-6)
#define TOL (1e-5)


void print_time (clock_t begin, const char *name)
{
    double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;
    if (time_spent >= 1e-1)
        printf("%s took %.4f s.\n", name, time_spent);
    else if (time_spent >= 1e-4)
        printf("%s took %.4f ms.\n", name, time_spent * 1e3);
    else if (time_spent >= 1e-7)
        printf("%s took %.4f us.\n", name, time_spent * 1e6);
    else
        printf("%s took %.4f ns.\n", name, time_spent * 1e9);
}


int test_kernel_grad (double x1, double x2, int npars, double *pars,
                      void *meta,
                      double (*kernel) (double, double, double*, void*, int,
                                        double*, int*))
{
    int i, count = 0, flag;
    double k0, kp, km, diff,
           *grad = malloc(npars * sizeof(double));

    k0 = (*kernel) (x1, x2, pars, meta, 1, grad, &flag);
    if (!flag) return 0;

    for (i = 0; i < npars; ++i) {
        pars[i] += EPS;
        kp = (*kernel) (x1, x2, pars, meta, 0, NULL, &flag);
        pars[i] -= 2 * EPS;
        km = (*kernel) (x1, x2, pars, meta, 0, NULL, &flag);
        pars[i] += EPS;

        diff = fabs((grad[i] - 0.5 * (kp - km) / EPS) / grad[i]);

        printf("    Parameter %d - analytical: %15.8e    " \
               "numerical: %15.8e    rel diff: %15.8e\t",
               i+1, grad[i], 0.5 * (kp - km) / EPS, diff);

        if (diff < TOL) {
            printf("[passed]\n");
            count += 1;
        } else printf("[failed]\n");
    }

    free (grad);

    return count;
}

int test_parameter_grad ()
{
    double pars[] = {1e-3, 3.0, 8.0};

    // Generate time series.
    int i, j, ndata = 1000, count = 0;
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
        } else printf("[failed]\n");
    }

    return count;
}

int main ()
{
    clock_t begin;
    double time_spent;

    int v, version[3];

#ifdef CHOLMOD_HAS_VERSION_FUNCTION
    cholmod_version (version);
    printf("CHOLMOD version: %d.%d.%d\n", version[0], version[1], version[2]);
#else
    v = CHOLMOD_VERSION;
    printf("CHOLMOD version: %d\n", v);
#endif

    printf("\nTesting kernel gradients\n");
    double pars[] = {1e-3, 3.0, 8.0};
    test_kernel_grad (0.3, 0.1, 3, pars, NULL, *george_kernel);

    printf("\nTesting parameter gradients\n");
    test_parameter_grad ();

    return 0;

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

    george_gp *gp = george_allocate_gp (3, pars, NULL, *george_kernel);
    printf("initialized\n");

    begin = clock();
    int info = george_compute (ndata, x, yerr, gp);
    print_time (begin, "compute");

    begin = clock();
    double lnlike = george_log_likelihood (y, gp);
    print_time (begin, "lnlike");

    for (i = 0, t = 0.0; i < ndata; ++i, t += 0.5 / 60.)
        y[i] = 5 * sin(t);

    begin = clock();
    lnlike = george_log_likelihood (y, gp);
    print_time (begin, "lnlike");

    free (x);
    free (yerr);
    george_free_gp(gp);
    return 0;
}
