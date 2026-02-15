# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "ELISA"
copyright = "2025, Jeison Steven Alfonso"
author = "Jeison Steven Alfonso"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- GitHub integration ------------------------------------------------------

html_context = {
    "display_github": True,
    "github_user": "stevenalfonso", 
    "github_repo": "elisa",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
