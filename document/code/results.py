


def prediction(times, lna, lns, lnd, fstar, q1, q2, t0, tau, ror, b, N=12):
    a, s = np.exp(lna), np.exp(lns)
    res = f - model(t, fstar, q1, q2, t0, tau, ror, b)
    gp = george.GaussianProcess([a, s], tol=1e-12, nleaf=40)
    gp.compute(t, fe)
    m2 = model(times, fstar, q1, q2, t0, tau, ror, b)
    samples = gp.sample_conditional(res, times, N=N)
    return samples + m2[None, :]


times = np.linspace(min(t), max(t), 500)
pl.plot(times, prediction(times, *p0).T, "k", alpha=0.5)
pl.savefig("initial.png")

