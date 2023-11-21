# -*- coding: utf-8 -*-

import os
import glob
import yaml
import george

# Inject the kernel docs
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(d, "docs", "user", "kernels.rst.template"), "r") as f:
    TEMPLATE = f.read()

fns = glob.glob(os.path.join(d, "kernels", "*.yml"))
if len(fns):
    specs = []
    for i, fn in enumerate(fns):
        with open(fn, "r") as f:
            specs.append(yaml.safe_load(f.read()))
    tokens = []
    for spec in specs:
        if spec["stationary"]:
            tokens += [
                ".. autoclass:: george.kernels.{0}".format(spec["name"])
            ]
    TEMPLATE = TEMPLATE.replace("STATIONARYKERNELS", "\n".join(tokens))
    tokens = []
    for spec in specs:
        if not spec["stationary"]:
            tokens += [
                ".. autoclass:: george.kernels.{0}".format(spec["name"])
            ]
    TEMPLATE = TEMPLATE.replace("OTHERKERNELS", "\n".join(tokens))

with open(os.path.join(d, "docs", "user", "kernels.rst"), "w") as f:
    f.write(TEMPLATE)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "george"
copyright = "2012-2023 Dan Foreman-Mackey"

version = george.__version__
release = george.__version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
html_theme = "pydata_sphinx_theme"
html_title = "george"
htmp_theme_options = dict(
    analytics_id="analytics_id",
)
# html_context = dict(
#     display_github=True,
#     github_user="dfm",
#     github_repo="george",
#     github_version="main",
#     conf_py_path="/docs/",
# )
html_static_path = ["_static"]
# html_show_sourcelink = False
