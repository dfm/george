#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "george.h"

#define EPS (1e-6)
#define TOL (1e-5)

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

int main ()
{
    double pars[] = {1e-3, 3.0, 8.0};
    int count = test_kernel_grad (0.3, 0.1, 3, pars, NULL, *george_kernel);
    if (count != 3) return -1;
    return 0;
}
