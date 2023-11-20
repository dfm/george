import os

import yaml
from jinja2 import Template


def compile_kernels(fns):
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
