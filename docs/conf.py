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
            specs.append(yaml.load(f.read()))
    tokens = []
    for spec in specs:
        if spec["stationary"]:
            tokens += [".. autoclass:: george.kernels.{0}"
                       .format(spec["name"])]
    TEMPLATE = TEMPLATE.replace("STATIONARYKERNELS", "\n".join(tokens))
    tokens = []
    for spec in specs:
        if not spec["stationary"]:
            tokens += [".. autoclass:: george.kernels.{0}"
                       .format(spec["name"])]
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
project = u"George"
copyright = u"2012-2018 Dan Foreman-Mackey"

version = george.__version__
release = george.__version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

htmp_theme_options = dict(
    analytics_id="analytics_id",
)
html_context = dict(
    display_github=True,
    github_user="dfm",
    github_repo="george",
    github_version="master",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
html_show_sourcelink = False
