#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["inject"]

import kplr
import fitsio
import transit
import numpy as np
import matplotlib.pyplot as pl


def inject(kicid, rng=6):
    # Download the data.
    client = kplr.API()
    kic = client.star(kicid)
    lcs = kic.get_light_curves(short_cadence=False)
    lc = lcs[np.random.randint(len(lcs))]

    # Read the data.
    data = lc.read()
    t = data["TIME"]
    f = data["SAP_FLUX"]
    fe = data["SAP_FLUX_ERR"]
    q = data["SAP_QUALITY"]

    # Remove missing points.
    m = np.isfinite(t) * np.isfinite(f) * np.isfinite(fe) * (q == 0)
    t, f, fe = t[m], f[m], fe[m]
    t -= t.min()

    # Build the transit system.
    s = transit.System(transit.Central(q1=np.random.rand(),
                                       q2=np.random.rand()))
    body = transit.Body(period=365.25, b=np.random.rand(), r=0.04,
                        t0=np.random.uniform(t.max()))
    s.add_body(body)

    # Compute the transit model.
    texp = kplr.EXPOSURE_TIMES[1] / 86400.0  # Long cadence exposure time.
    model = s.light_curve(t, texp=texp)
    f *= model

    # Trim the dataset to include data only near the transit.
    m = np.abs(t - body.t0) < rng
    t, f, fe = t[m], f[m], fe[m]
    t -= body.t0

    # Save the injection as a FITS light curve.
    dt = [("TIME", float), ("SAP_FLUX", float), ("SAP_FLUX_ERR", float)]
    data = np.array(zip(t, f, fe), dtype=dt)
    hdr = dict(b=body.b, period=body.period, r=body.r, t0=0.0,
               q1=s.central.q1, q2=s.central.q2)
    fitsio.write("{0}-injection.fits".format(kicid), data, header=hdr,
                 clobber=True)

    # Plot the light curve.
    ppm = (f / np.median(f) - 1) * 1e6
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, ppm, ".k")
    ax.set_xlim(-rng, rng)
    ax.set_xlabel("time since transit [days]")
    ax.set_ylabel("relative flux [ppm]")
    ax.set_title("raw light curve")
    fig.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    fig.savefig("{0}-raw.pdf".format(kicid))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        inject(sys.argv[1])
    else:
        np.random.seed(12345)
        inject(2301306)
        # inject(2973073)
