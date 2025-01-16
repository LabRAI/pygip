import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Project information
project = 'PyGIP'
copyright = '2024, Yuxiang Sun, Chenxi Zhao'
author = 'Yuxiang Sun, Chenxi Zhao'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.githubpages'  # Added for GitHub Pages support
]

templates_path = ['_templates']
exclude_patterns = []

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'show-inheritance': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# HTML theme settings
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [],
    "light_css_variables": {
        "color-foreground-primary": "black",
        "color-background-primary": "white",
        "color-background-secondary": "#f8f9fb",
    }
}
# Disable view source link
html_show_sourcelink = False

# Important for GitHub Pages
html_baseurl = 'https://labrai.github.io/pygip/'
