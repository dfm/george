#!/usr/bin/env python

import os

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    setup, Extension

import numpy


gp_ext = Extension("george._gp",
                sources=["src/gp.cpp", "src/kernels.cpp", "src/python-gp.cpp"],
                include_dirs=["src", numpy.get_include(),
                    os.environ.get("EIGEN_DIR", "/usr/local/include/eigen3")],
                )

setup(
    name="george",
    version="0",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["george"],
    ext_modules=[gp_ext],
    description="Blazingly fast Gaussian Processes.",
)
