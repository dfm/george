#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_data"]

import fitsio
import numpy as np


def median_trend(x, y, dt):
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = np.abs(x-t) < 0.5 * dt
        r[i] = np.median(y[inds])
    return r


def load_data(fn, median=False, dt=3.0):
    data, hdr = fitsio.read(fn, ext=1, header=True)
    t, f, fe = data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"]

    # Median detrend if requested.
    if median:
        trend = median_trend(t, f, dt)
        f /= trend
        fe /= trend

    # Parse the true values.
    truth = dict((k.lower(), hdr[k]) for k in ["Q1", "Q2", "R", "B", "PERIOD",
                                               "T0"])

    return t, f, fe, truth


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as pl

    # Load and detrend the data.
    t, f, fe, truth = load_data(sys.argv[1], True)

    # Plot the detrended light curve.
    ppm = (f / np.median(f) - 1) * 1e6
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, ppm, ".k")
    ax.set_xlim(np.min(t), np.max(t))
    ax.set_xlabel("time since transit [days]")
    ax.set_ylabel("relative flux [ppm]")
    ax.set_title("median de-trended light curve")
    fig.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    fig.savefig(os.path.splitext(sys.argv[1])[0] + "-median.pdf")
