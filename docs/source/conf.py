# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Config file for docs.
"""

import os.path as op
import sys

sys.path.insert(0, op.join(op.dirname(__file__), "../../", "src"))

import dxtb

project = "Fully Differentiable Extended Tight-Binding"
author = "Grimme Group"
copyright = f"2024 {author}"

extensions = [
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",  # after autodoc
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/dxtb.svg"
html_favicon = "_static/dxtb-favicon.svg"

html_theme_options = {
    "repository_url": "https://github.com/grimme-lab/dxtb",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "path_to_docs": "docs/source",
    "show_navbar_depth": 3,
}

html_sidebars = {}  # type: ignore[var-annotated]

html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]

autodoc_typehints = "none"
autosummary_generate = True
autosummary_imported_members = True

# Configuration for sphinx-copybutton
copybutton_prompt_text = ">>> |... "
copybutton_prompt_is_regexp = True

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

master_doc = "index"
