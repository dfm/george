#!/usr/bin/env python

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    setup, Extension


from numpy.distutils.misc_util import get_numpy_include_dirs


gp_ext = Extension("gp._gp",
                sources=["src/gp.cpp", "src/_gp.cpp"],
                include_dirs=["include", "src"] + get_numpy_include_dirs(),
                )

setup(
    name="gp",
    version="0.0.1",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["gp"],
    ext_modules=[gp_ext],
    description="Gaussian Processes.",
)
