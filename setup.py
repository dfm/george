#!/usr/bin/env python

import os
import sys

if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    setup, Extension

import numpy

include_dirs = [
    "george",
    numpy.get_include(),
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    "/usr/local/homebrew/include",
    "/usr/local/homebrew/include/eigen3",
    "/opt/local/var/macports/software/eigen3/opt/local/include/eigen3",
]

sources = [
    os.path.join("george", "_george.cpp"),
]

gp_ext = Extension("george._george",
                   sources=sources,
                   include_dirs=include_dirs)

setup(
    name="george",
    version="0.2.0",
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    url="https://github.com/dfm/george",
    packages=["george"],
    ext_modules=[gp_ext],
    description="Blazingly fast Gaussian Processes.",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE.rst"]},
    include_package_data=True,
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
