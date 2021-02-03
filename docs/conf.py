# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import atexit
import glob
import os
import re
import sys

import guzzle_sphinx_theme
from setuptools_scm import get_version

sys.path.insert(0, os.path.abspath(".."))

__version__ = get_version(root="..", relative_to=__file__)
print("__version__", __version__)

with open("./_templates/version.html", "w") as f:
    f.write('<div align="center">v%s</div>' % __version__)


# -- Project information -----------------------------------------------------

project = "QuantumFlow"
copyright = "2019, Gavin Crooks"
author = "Gavin Crooks"
release = __version__

html_title = "QuantumFlow Documentation"
html_short_title = "QuantumFlow"

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "**": ["logo-text.html", "version.html", "searchbox.html", "globaltoc.html"]
}


def setup(app):
    app.add_css_file("qf.css")  # also can be a full URL


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    #  'sphinx_autodoc_typehints',
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
]

extensions.append("guzzle_sphinx_theme")

bibtex_bibfiles = ["references.bib"]

napoleon_use_ivar = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def do_edits():

    pats = [
        # Hacks to shorten type descriptors
        (r"quantumflow\.qubits", r"qf"),
        (r"quantumflow\.ops", r"qf"),
        (r"quantumflow\.stdops", r"qf"),
        (r"quantumflow\.gates\.gates_one", r"qf"),
        (r"quantumflow\.gates\.gates_two", r"qf"),
        (r"quantumflow\.gates\.gates_three", r"qf"),
        (r"quantumflow\.gates\.gates_qasm", r"qf"),
        (r"quantumflow\.gates", r"qf"),
        (r"quantumflow\.states", r"qf"),
        (r"quantumflow\.circuits", r"qf"),
        (r"quantumflow\.translate\.", r"qf."),
        (r"^/quantumflow\.", r"qf."),
        # (r'qf\.readthedocs\.io', r'quantumflow.readthedocs.io'),
        # Hacks to fixup types
        # (r"sympy\.core\.symbol\.Symbol", r"Parameter"),
        # (r"Sequence\[collections\.abc\.Hashable\]", "Qubits"),
        # (r"collections\.abc\.Hashable", "Qubit"),
        (r"tensor: Any", r"tensor: TensorLike"),
        (r"&#x2192; Any", r"&#x2192; BKTensor"),
        ("Sequence[qf.Qubit]", "qf.Qubits"),
        ("Sequence[Union[float, sympy.core.expr.Expr]]", "qf.Variable"),
    ]

    files = glob.glob("*build/html/*.html")
    for filename in files:
        with open(filename, "r+") as f:
            text = f.read()
            for pattern, replacement in pats:
                text = re.sub(pattern, replacement, text)
            f.seek(0)
            f.truncate()
            f.write(text)

    print("Note: post sphinx text substitutions performed (conf.py)")


atexit.register(do_edits)
