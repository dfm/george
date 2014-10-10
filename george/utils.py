# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["multivariate_gaussian_samples", "nd_sort_samples"]

import numpy as np
from scipy.spatial import cKDTree


def multivariate_gaussian_samples(matrix, N, mean=None):
    """
    Generate samples from a multidimensional Gaussian with a given covariance.

    :param matrix: ``(k, k)``
        The covariance matrix.

    :param N:
        The number of samples to generate.

    :param mean: ``(k,)`` (optional)
        The mean of the Gaussian. Assumed to be zero if not given.

    :returns samples: ``(k,)`` or ``(N, k)``
        Samples from the given multivariate normal.

    """
    if mean is None:
        mean = np.zeros(len(matrix))
    samples = np.random.multivariate_normal(mean, matrix, N)
    if N == 1:
        return samples[0]
    return samples


def nd_sort_samples(samples):
    """
    Sort an N-dimensional list of samples using a KDTree.

    :param samples: ``(nsamples, ndim)``
        The list of samples. This must be a two-dimensional array.

    :returns i: ``(nsamples,)``
        The list of indices into the original array that return the correctly
        sorted version.

    """
    # Check the shape of the sample list.
    assert len(samples.shape) == 2

    # Build a KD-tree on the samples.
    tree = cKDTree(samples)

    # Compute the distances.
    d, i = tree.query(samples[0], k=len(samples))
    return i


def numerical_gradient(f, x, dx=1.234e-6):
    g = np.empty_like(x, dtype=float)
    for i in range(len(g)):
        x[i] += dx
        fp = f(x)
        x[i] -= 2*dx
        fm = f(x)
        x[i] += dx
        g[i] = 0.5 * (fp - fm) / dx
    return g
