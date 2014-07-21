#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["results"]

import os
import triangle
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as pl


def results(fn):
    model, sampler = pickle.load(open(fn, "rb"))
    mu = np.median(model.f)
    ppm = lambda f: (f / mu - 1) * 1e6

    # Plot the data.
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(model.t, ppm(model.f), ".k")
    ax.set_xlim(np.min(model.t), np.max(model.t))
    ax.set_xlabel("time since transit [days]")
    ax.set_ylabel("relative flux [ppm]")
    fig.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)

    # Plot the predictions.
    samples = sampler.flatchain
    t = np.linspace(model.t.min(), model.t.max(), 1000)
    for i in np.random.randint(len(samples), size=10):
        model.vector = samples[i]
        ax.plot(t, ppm(model.predict(t)), color="#4682b4", alpha=0.5)

    fig.savefig(os.path.splitext(fn)[0] + "-results.pdf")

    # Plot the corner plot.
    fig = triangle.corner(samples, labels=model.labels,
                          truths=model.true_vector)
    fig.savefig(os.path.splitext(fn)[0] + "-triangle.png")


if __name__ == "__main__":
    import sys
    results(sys.argv[1])
