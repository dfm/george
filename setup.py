#!/usr/bin/env python

import os
import re


def find_eigen(hint=None):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.

    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else [hint]
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None


def find_hodlr(hint=None):
    """
    Find the location of the HODLR include directory. This will return
    ``None`` on failure.

    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else [hint]
    search_dirs += [
        "./hodlr/header",
        "/usr/local/include",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        paths = [
            os.path.join(d, "HODLR_Matrix.hpp"),
            os.path.join(d, "HODLR_Tree.hpp"),
            os.path.join(d, "HODLR_Node.hpp"),
        ]
        if all(map(os.path.exists, paths)):
            print("Found HODLR headers in: {0}".format(d))
            return d
    return None


if __name__ == "__main__":
    import sys
    import numpy
    import argparse
    from setuptools import setup, Extension

    # Publish the library to PyPI.
    if "publish" in sys.argv[-1]:
        os.system("python setup.py sdist upload")
        sys.exit()

    # Allow the user to specify custom search locations.
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-include", dest="eigen",
                        help="Path to Eigen include directory")
    parser.add_argument("--hodlr-include", dest="hodlr",
                        help="Path to HODLR include directory")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    # Find the Eigen include directory.
    eigen_include = find_eigen(hint=args.eigen)
    if eigen_include is None:
        raise RuntimeError("Required library Eigen 3 not found. "
                           "Try specifying the --eigen-include=\"...\" option "
                           "at the command line.")

    # Find the HODLR include directory.
    hodlr_include = find_hodlr(hint=args.hodlr)
    if hodlr_include is None:
        raise RuntimeError("Required library HODLR not found. "
                           "Try specifying the --hodlr-include=\"...\" option "
                           "at the command line.")

    # Set up the C++-extension.
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    include_dirs = [
        "include",
        numpy.get_include(),
        eigen_include,
        hodlr_include,
    ]
    ext = Extension("george._george", sources=["george/_george.cc"],
                    libraries=libraries, include_dirs=include_dirs)

    # Hackishly inject a constant into builtins to enable importing of the
    # package before the library is built.
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins
    builtins.__GEORGE_SETUP__ = True
    import george

    setup(
        name="george",
        version=george.__version__,
        author="Daniel Foreman-Mackey",
        author_email="danfm@nyu.edu",
        url="https://github.com/dfm/george",
        packages=["george"],
        ext_modules=[ext],
        description="Blazingly fast Gaussian Processes for regression.",
        long_description=open("README.rst").read(),
        package_dir={"": ""},
        classifiers=[
            # "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
        ],
    )
