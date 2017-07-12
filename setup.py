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

    with open(os.path.join(template_dir, "parser.h")) as f:
        PARSER_TEMPLATE = Template(f.read())
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

    fn = os.path.join(output_dir, "include", "george", "parser.h")
    with open(fn, "w") as f:
        print("Saving parser to '{0}'".format(fn))
        f.write(PARSER_TEMPLATE.render(specs=specs))
    fn = os.path.join(output_dir, "include", "george", "kernels.h")
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

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class build_ext(_build_ext):
    c_opts = {
        "msvc": ["/EHsc", "/DNODEBUG"],
        "unix": ["-DNODEBUG"],
    }

    def build_extensions(self):
        # The include directory for the celerite headers
        localincl = "vendor"
        if not os.path.exists(os.path.join(localincl, "eigen_3.3.4", "Eigen",
                                           "Core")):
            raise RuntimeError("couldn't find Eigen headers")

        # Add the pybind11 include directory
        import numpy
        import pybind11
        include_dirs = [
            os.path.join("george", "include"),
            os.path.join(localincl, "eigen_3.3.4"),
            numpy.get_include(),
            pybind11.get_include(False),
            pybind11.get_include(True),
        ]
        for ext in self.extensions:
            ext.include_dirs = include_dirs + ext.include_dirs

        # Compiler flags
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-DVERSION_INFO=\"{0:s}\""
                        .format(self.distribution.get_version()))
            print("testing C++14/C++11 support")
            opts.append(cpp_flag(self.compiler))

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

    extensions = [
        Extension("george.kernel_interface",
                  sources=[os.path.join("george", "kernel_interface.cpp")],
                  language="c++"),
        Extension("george.solvers.hodlr",
                  sources=[os.path.join("george", "solvers", "hodlr.cpp")],
                  language="c++"),
    ]

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
        package_data={"": ["README.rst", "LICENSE", "AUTHORS.rst",
                           "HISTORY.rst"]},
        install_requires=["numpy", "scipy", "pybind11"],
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
        zip_safe=True,
    )
