#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["process_specs"]

import os
import yaml
from jinja2 import Template

basepath = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(basepath, "templates")
output_dir = os.path.join(basepath, "george")

with open(os.path.join(template_dir, "kerneldefs.pxd")) as f:
    PXD_TEMPLATE = Template(f.read())
with open(os.path.join(template_dir, "kernels.h")) as f:
    CPP_TEMPLATE = Template(f.read())
with open(os.path.join(template_dir, "kernels.py")) as f:
    PYTHON_TEMPLATE = Template(f.read())


def process_specs(fns):
    specs = []
    for i, fn in enumerate(fns):
        with open(fn, "r") as f:
            spec = yaml.load(f.read())
        spec["index"] = i
        specs.append(spec)

    with open(os.path.join(output_dir, "kerneldefs.pxd"), "w") as f:
        f.write(PXD_TEMPLATE.render(specs=specs))
    with open(os.path.join(output_dir, "include", "kernels.h"), "w") as f:
        f.write(CPP_TEMPLATE.render(specs=specs))
    with open(os.path.join(output_dir, "kernels.py"), "w") as f:
        f.write(PYTHON_TEMPLATE.render(specs=specs))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="a list of files to process")
    args = parser.parse_args()
    process_specs(args.files)
