#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as pl

d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
import george
from george.kernels import ExpSquaredKernel, Matern32Kernel, CosineKernel

np.random.seed(12345)

experiments = [
    ("exponential squared", [
        ("-k", "$l=0.5$", ExpSquaredKernel(0.5)),
        ("--k", "$l=1$", ExpSquaredKernel(1.0)),
        (":k", "$l=2$", ExpSquaredKernel(2.0)),
    ]),
    ("quasi-periodic", [
        ("-k", "$l=2,\,P=3$", Matern32Kernel(2.0) * CosineKernel(3.0)),
        ("--k", "$l=3,\,P=3$", Matern32Kernel(3.0) * CosineKernel(3.0)),
        (":k", "$l=3,\,P=1$", Matern32Kernel(3.0) * CosineKernel(1.0)),
    ])
]

t = np.linspace(0, 10, 500)
h, w = len(experiments) * 4, 6
fig, axes = pl.subplots(len(experiments), 1, figsize=(w, h), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.98,
                    wspace=0.0, hspace=0.05)
for ax, (name, runs) in zip(axes, experiments):
    for style, label, kernel in runs:
        gp = george.GP(kernel)
        f = gp.sample(t)
        ax.plot(t, f, style, lw=1.5, label=label)
    ax.legend(prop={"size": 12})
    ax.set_ylim(-2.8, 3.8)
    ax.annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(5, -5),
                textcoords="offset points", ha="left", va="top")
axes[-1].set_xlabel("$t$")
fig.savefig("simple.png")
fig.savefig("simple.pdf")

# Plot the covariance matrix images.
fig = pl.figure(figsize=(8, 8))
for i, (name, runs) in enumerate(experiments):
    fig.clf()
    gp = george.GP(runs[0][2])
    img = gp.get_matrix(t)
    ax = fig.add_subplot(111)
    ax.set_title(name)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.savefig("simple-cov-{0}.pdf".format(i+1))
