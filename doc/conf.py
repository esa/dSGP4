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


# -- Project information -----------------------------------------------------

project = "dsgp4"
copyright = "2022, 2023, 2024, 2025, Giacomo Acciarini and Atılım Güneş Baydin and Dario Izzo"
author = "Giacomo Acciarini, Atılım Güneş Baydin, Dario Izzo"

# The full version, including alpha/beta/rc tags
import dsgp4
import sys
import os
sys.path.insert(0, os.path.abspath('../'))  # Add the root directory of your repo

release = dsgp4.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.intersphinx", "sphinx.ext.autosummary","sphinx.ext.napoleon"]


# build the templated autosummary files
autosummary_generate = True
autosummary_imported_members = True
napoleon_google_docstring = True
numpydoc_show_class_members = False
panels_add_bootstrap_css = False

autosectionlabel_prefix_document = True

# katex options
#
#
katex_prerender = True

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

autoclass_content = 'both'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", ".DS_Store",'jupyter_execute/**/*.ipynb','jupyter_execute/*.ipynb']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/logo_dsgp4.png"


linkcheck_ignore = [
     r'https://www\.esa\.int/gsp/ACT/team/giacomo_acciarini/',
     r'https://www.esa.int/gsp/ACT/team/dario_izzo/',
 ]
 

html_theme_options = {
    "repository_url": "https://github.com/esa/dSGP4/",
    "repository_branch": "master",
    "path_to_docs": "doc",
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab"
    },
    "navigation_with_keys": False,
}

nb_execution_mode = "force"

nb_execution_excludepatterns = [
    "tle_propagation.ipynb",
    "covariance_propagation.ipynb",
    "mldsgp4.ipynb"
]
autosummary_ignore_module = ['dsgp4.mldsgp4']

latex_engine = "xelatex"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
