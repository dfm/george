import numpy as np
import matplotlib.pyplot as pl
from george.kernels import ExpSquaredKernel
r = np.linspace(0, 3)
kr = ExpSquaredKernel(1.0).get_value(r * r)
pl.plot(r, kr, "k", lw=2)
pl.show()