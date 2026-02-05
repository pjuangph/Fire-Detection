"""Sphinx configuration for Fire-Detection documentation."""

import os
import sys

# Add project root to sys.path so Sphinx can import lib/
sys.path.insert(0, os.path.abspath('..'))

project = 'Fire-Detection'
copyright = '2024, Team Flaming Kitty'
author = 'Team Flaming Kitty'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

# Napoleon: existing docstrings use Google-style (Args: / Returns:)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = False
napoleon_include_init_with_doc = True

# Autodoc
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Mock heavy/system-dependent imports for CI builds
autodoc_mock_imports = ['pyhdf', 'torch', 'scipy', 'sklearn', 'earthaccess']

# Intersphinx: link to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Theme
html_theme = 'furo'
html_title = 'MASTER Fire Detection'
html_static_path = ['_static']
html_css_files = ['custom.css']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
