#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A reproduction of Figure 5.6 from Rasmussen & Williams (2006).
http://www.gaussianprocess.org/gpml/

"""

from __future__ import division, print_function

import numpy as np
import scipy.optimize as op
import statsmodels.api as sm
import matplotlib.pyplot as pl

import george
from george import kernels

# Load the dataset.
data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)
base = np.mean(y)

# Plot the data.
fig = pl.figure(figsize=(6, 3.5))
ax = fig.add_subplot(111)
ax.plot(t, y, ".k", ms=2)
ax.set_xlim(min(t), 1999)
ax.set_ylim(min(y), 369)
ax.set_xlabel("year")
ax.set_ylabel("CO$_2$ in ppm")
fig.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.95)
fig.savefig("../_static/hyper/data.png", dpi=150)
fig.savefig("hyper-data.pdf")

# Initialize the kernel.
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) \
    * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
kernel = k1 + k2 + k3 + k4

# Set up the Gaussian process and maximize the marginalized likelihood.
gp = george.GP(kernel, mean=np.mean(y))

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(y, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    return -gp.grad_lnlikelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t)

# Print the initial ln-likelihood.
print(gp.lnlikelihood(y))

# Run the optimization routine.
p0 = gp.kernel.vector
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.kernel[:] = results.x
print(results)
p = gp.kernel.pars
print("Final parameters: \n", p)
print("Final marginalized ln-likelihood: {0}".format(gp.lnlikelihood(y)))

# Build the results table.
rw = [66**2, 67**2, 2.4**2, 90**2, 2.0 / 1.3**2, 1.0, 0.66**2, 0.78, 1.2**2,
      0.18**2, 1.6**2, 0.19]
labels = map(r":math:`\theta_{{{0}}}`".format, range(1, len(p)+1))
l = max(map(len, labels))
rf = "| {{0[0]:{0}s}} | {{0[1]:10.2f}} | {{0[2]:10.2f}} |".format(l).format
header = "| {{0:{0}s}} | {{1:10s}} | {{2:10s}} |".format(l).format(
    "", "result", "R&W")
sep = "+" + "+".join(["-" * (l+2), "-" * 12, "-" * 12]) + "+"
rows = ("\n"+sep+"\n").join(map(rf, zip(labels, p, rw)))
table = "\n".join([sep, header, sep.replace("-", "="), rows, sep])
with open("../_static/hyper/results.txt", "w") as f:
    f.write(table)

# Compute the prediction into the future.
x = np.linspace(max(t), 2025, 2000)
mu, cov = gp.predict(y, x)
std = np.sqrt(np.diag(cov))

# Plot the prediction.
ax.fill_between(x, mu+std, mu-std, color="k", alpha=0.4)
ax.set_xlim(min(t), 2025.0)
ax.set_ylim(min(y), 400.0)
fig.savefig("../_static/hyper/figure.png", dpi=150)
fig.savefig("hyper-figure.pdf", dpi=150)
