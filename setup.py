#!/usr/bin/env python

import os
import tempfile

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def compile_kernels(fns):
    import yaml
    from jinja2 import Template

    template_dir = "templates"
    output_dir = "george"

    with open(os.path.join(template_dir, "kerneldefs.pxd")) as f:
        PXD_TEMPLATE = Template(f.read())
    with open(os.path.join(template_dir, "kernels.h")) as f:
        CPP_TEMPLATE = Template(f.read())
    with open(os.path.join(template_dir, "kernels.py")) as f:
        PYTHON_TEMPLATE = Template(f.read())

    specs = []
    for i, fn in enumerate(fns):
        with open(fn, "r") as f:
            spec = yaml.load(f.read())
        print("Found kernel '{0}'".format(spec["name"]))
        spec["index"] = i
        spec["reparams"] = spec.get("reparams", {})
        specs.append(spec)
    print("Found {0} kernel specifications".format(len(specs)))

    fn = os.path.join(output_dir, "kerneldefs.pxd")
    with open(fn, "w") as f:
        print("Saving Cython kernels to '{0}'".format(fn))
        f.write(PXD_TEMPLATE.render(specs=specs))
    fn = os.path.join(output_dir, "solvers", "kerneldefs.pxd")
    with open(fn, "w") as f:
        print("Saving Cython kernels to '{0}'".format(fn))
        f.write(PXD_TEMPLATE.render(specs=specs))
    fn = os.path.join(output_dir, "include", "kernels.h")
    with open(fn, "w") as f:
        print("Saving C++ kernels to '{0}'".format(fn))
        f.write(CPP_TEMPLATE.render(specs=specs))
    fn = os.path.join(output_dir, "kernels.py")
    with open(fn, "w") as f:
        print("Saving Python kernels to '{0}'".format(fn))
        f.write(PYTHON_TEMPLATE.render(specs=specs))

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def has_library(compiler, libname):
    """Return a boolean indicating whether a library is found."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True

class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Eigen
    and HODLR before compiling.

    """
    c_opts = {
        "msvc": ["/EHsc", "/DNODEBUG"],
        "unix": ["-DNODEBUG"],
    }

    def build_extensions(self):
        # The include directory for the celerite headers
        localincl = "vendor"
        if not os.path.exists(os.path.join(localincl, "hodlr", "header",
                                           "HODLR_Matrix.hpp")):
            raise RuntimeError("couldn't find HODLR headers")
        if not os.path.exists(os.path.join(localincl, "eigen_3.3.4", "Eigen",
                                           "Core")):
            raise RuntimeError("couldn't find Eigen headers")

        # Add the pybind11 include directory
        import numpy
        include_dirs = [
            "george",
            os.path.join("george", "include"),
            os.path.join(localincl, "hodlr", "header"),
            os.path.join(localincl, "eigen_3.3.4"),
            numpy.get_include(),
        ]
        for ext in self.extensions:
            ext.include_dirs = include_dirs + ext.include_dirs

        # Compiler flags
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-DVERSION_INFO=\"{0:s}\""
                        .format(self.distribution.get_version()))

            flags = ["-stdlib=libc++", "-funroll-loops",
                     "-Wno-unused-function", "-Wno-uninitialized",
                     "-Wno-unused-local-typedefs"]

            # Mac specific flags and libraries
            if sys.platform == "darwin":
                flags += ["-march=native", "-mmacosx-version-min=10.9"]
                for lib in ["m", "c++"]:
                    for ext in self.extensions:
                        ext.libraries.append(lib)
                for ext in self.extensions:
                    ext.extra_link_args += ["-mmacosx-version-min=10.9",
                                            "-march=native"]
            else:
                libraries = ["m", "stdc++", "c++"]
                for lib in libraries:
                    if not has_library(self.compiler, lib):
                        continue
                    for ext in self.extensions:
                        ext.libraries.append(lib)

            # Check the flags
            print("testing compiler flags")
            for flag in flags:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

        elif ct == "msvc":
            opts.append("/DVERSION_INFO=\\\"{0:s}\\\""
                        .format(self.distribution.get_version()))

        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Run the standard build procedure.
        _build_ext.build_extensions(self)


if __name__ == "__main__":
    import sys
    import glob

    # Publish the library to PyPI.
    if "publish" in sys.argv[-1]:
        os.system("python setup.py sdist upload")
        sys.exit()

    # If the kernel specifications are included (development mode) re-compile
    # them first.
    kernel_specs = glob.glob(os.path.join("kernels", "*.yml"))
    if len(kernel_specs):
        print("Compiling kernels")
        compile_kernels(kernel_specs)
        if "kernels" in sys.argv:
            sys.exit()

    # Check for the Cython source (development mode) and compile it if it
    # exists.
    kern_fn = os.path.join("george", "cython_kernel")
    hodlr_fn = os.path.join("george", "solvers", "hodlr")
    if (os.path.exists(kern_fn + ".pyx") and
            os.path.exists(hodlr_fn + ".pyx") and
            os.path.exists(os.path.join("george", "kerneldefs.pxd"))):
        from Cython.Build import cythonize
        kern_fn += ".pyx"
        hodlr_fn += ".pyx"
    else:
        kern_fn += ".cpp"
        hodlr_fn += ".cpp"
        cythonize = lambda x: x

    kern_ext = Extension("george.cython_kernel", sources=[kern_fn])
    hodlr_ext = Extension("george.solvers.hodlr", sources=[hodlr_fn])
    extensions = cythonize([kern_ext, hodlr_ext])

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
        author_email="foreman.mackey@gmail.com",
        url="https://github.com/dfm/george",
        license="MIT",
        packages=["george", "george.solvers"],
        ext_modules=extensions,
        description="Blazingly fast Gaussian Processes for regression.",
        long_description=open("README.rst").read(),
        package_data={
            "": ["README.rst", "LICENSE",
                 os.path.join("george", "include", "*.h"),
                 os.path.join("vendor", "hodlr", "header", "*.hpp")]
        },
        include_package_data=True,
        cmdclass=dict(build_ext=build_ext),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
        ],
    )
