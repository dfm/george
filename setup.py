#!/usr/bin/env python

import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext


def compile_kernels(fns):
    import yaml
    from jinja2 import Template

    template_dir = "templates"
    output_dir = os.path.join("src", "george")

    with open(os.path.join(template_dir, "parser.h")) as f:
        PARSER_TEMPLATE = Template(f.read())
    with open(os.path.join(template_dir, "kernels.h")) as f:
        CPP_TEMPLATE = Template(f.read())
    with open(os.path.join(template_dir, "kernels.py")) as f:
        PYTHON_TEMPLATE = Template(f.read())

    specs = []
    for i, fn in enumerate(fns):
        with open(fn, "r") as f:
            spec = yaml.load(f.read(), Loader=yaml.FullLoader)
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


if __name__ == "__main__":
    # The include directory for the Eigen headers
    localincl = "vendor"
    if not os.path.exists(os.path.join(localincl, "eigen", "Eigen", "Core")):
        raise RuntimeError("couldn't find Eigen headers")

    include_dirs = [
        os.path.join("src", "george", "include"),
        os.path.join(localincl, "eigen"),
    ]
    extensions = [
        Pybind11Extension(
            "george.kernel_interface",
            sources=[os.path.join("src", "george", "kernel_interface.cpp")],
            language="c++",
            include_dirs=include_dirs,
        ),
        Pybind11Extension(
            "george.solvers._hodlr",
            sources=[os.path.join("src", "george", "solvers", "_hodlr.cpp")],
            language="c++",
            include_dirs=include_dirs,
        ),
    ]

    setup(
        name="george",
        author="Daniel Foreman-Mackey",
        author_email="foreman.mackey@gmail.com",
        url="https://github.com/dfm/george",
        license="MIT",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=extensions,
        description="Blazingly fast Gaussian Processes for regression.",
        long_description=open("README.rst").read(),
        package_data={
            "": ["README.rst", "LICENSE", "AUTHORS.rst", "HISTORY.rst"]
        },
        install_requires=["numpy", "scipy"],
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
