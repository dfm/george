#ifndef _KERNELS_H_
#define _KERNELS_H_

double isotropicKernel (double x1, double x2, int npars, double *pars);
void gradIsotropicKernel (double x1, double x2, int npars, double *pars,
                          double *dkdt);

#endif
// /_KERNELS_H_
