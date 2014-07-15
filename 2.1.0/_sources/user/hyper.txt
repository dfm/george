.. _hyper:

Tutorial: setting the hyperparameters
=====================================

In this demo, we'll reproduce the analysis for Figure 5.6 in `Chapter 5 of
Rasmussen & Williams (R&W)
<http://www.gaussianprocess.org/gpml/chapters/RW5.pdf>`_.
The data are measurements of the atmospheric CO2 concentration made at Mauna
Loa, Hawaii (Keeling & Whorf 2004).
The dataset is said to be available online but I couldn't seem to download it
from the original source.
Luckily the `statsmodels <http://statsmodels.sourceforge.net/>`_ package
`includes a copy
<http://statsmodels.sourceforge.net/devel/datasets/generated/co2.html>`_ that
we can load as follows:

.. code-block:: python

    import numpy as np
    import statsmodels.api as sm

    data = sm.datasets.get_rdataset("co2").data
    t = np.array(data.time)
    y = np.array(data.co2)

These data are plotted in the figure below:

.. image:: ../_static/hyper/data.png

In this figure, you can see that there is periodic (or quasi-periodic) signal
with a year-long period superimposed on a long term trend.
We will follow R&W and model these effects non-parametrically using a
complicated covariance function.
The covariance function that we'll use is:

.. math::

    k(r) = k_1(r) + k_2(r) + k_3(r) + k_4(r)

where

.. math::

    \begin{eqnarray}
    k_1(r) &=& \theta_1^2 \, \exp \left(-\frac{r^2}{2\,\theta_2} \right) \\
    k_2(r) &=& \theta_3^2 \, \exp \left(-\frac{r^2}{2\,\theta_4}
                                         -\theta_5\,\sin^2\left(
                                         \frac{\pi\,r}{\theta_6}\right)
                                        \right) \\
    k_3(r) &=& \theta_7^2 \, \left [ 1 + \frac{r^2}{2\,\theta_8\,\theta_9}
                             \right ]^{-\theta_8} \\
    k_4(r) &=& \theta_{10}^2 \, \exp \left(-\frac{r^2}{2\,\theta_{11}} \right)
                + \theta_{12}^2\,\delta_{ij}
    \end{eqnarray}

We can implement this kernel in George as follows (we'll use the R&W results
as the hyperparameters for now):

.. code-block:: python

    from george import kernels

    k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
    k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
    k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
    k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
    kernel = k1 + k2 + k3 + k4

Then, to find the "best-fit" hyperparameters, we want to maximize the
ln-likelihood as a function of the hyperparameters:

.. code-block:: python

    import george
    gp = george.GP(kernel, mean=np.mean(y))
    gp.compute(t)
    print(gp.lnlikelihood(y))

In general, you'll probably want to write a custom routine for optimizing this
function---possibly using the gradient computed using
:func:`george.GP.grad_lnlikelihood`---but George does come with a simple
gradient-based non-linear optimization routine that isn't a bad starting
point:

.. code-block:: python

    p, results = gp.optimize(t, y)

Running this optimization, we find a final ln-likelihood of -100.22 (slightly
better than the result in R&W) and the following parameter values:

.. include:: ../_static/hyper/results.txt

We can plot our prediction of the CO2 concentration into the future using our
optimized Gaussian process model by running:

.. code-block:: python

    x = np.linspace(max(t), 2025, 2000)
    mu, cov = gp.predict(y, x)
    std = np.sqrt(np.diag(cov))

and this gives a result just like Figure 5.6 from R&W:

.. image:: ../_static/hyper/figure.png
