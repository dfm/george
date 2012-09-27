#!/usr/bin/env python

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    setup, Extension


gp_ext = Extension("gp._gp",
                   sources=["src/_gp.cpp", "src/gp.cpp"],
                   libraries=["stdc++"])

setup(
    name="gp",
    version="0.0.1",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["gp"],
    ext_modules=[gp_ext],
    description="Gaussian Processes.",
    include_dirs=["include", "src"],
)
