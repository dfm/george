# -*- coding: utf-8 -*-

import os
import sys

d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
import george

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# # FIXME
# mathjax_path = "MathJax/MathJax.js"

# General information about the project.
project = u"George"
copyright = u"2013-2014 Dan Foreman-Mackey"

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
    script_files=[
        "_static/jquery.js",
        "_static/underscore.js",
        "_static/doctools.js",
        "//cdn.mathjax.org/mathjax/latest/MathJax.js"
        "?config=TeX-AMS-MML_HTMLorMML",
        "_static/js/analytics.js",
    ],
)
html_static_path = ["_static"]
html_show_sourcelink = False
