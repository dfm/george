#!/usr/bin/env python

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    setup, Extension


import numpy


gp_ext = Extension("george._gp",
                sources=["src/gp.cpp", "src/python-gp.cpp"],
                include_dirs=["src", numpy.get_include()],
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
