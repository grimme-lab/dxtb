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
Test the lazy loaders.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dxtb._src.loader.lazy import attach_module


def test_attach_module_imports_submodules():
    package_name = "test_package"
    submodules = ["sub1", "sub2"]

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_import_module.side_effect = lambda name: f"module_{name}"

        __getattr__, __dir__, __all__ = attach_module(package_name, submodules)

        # Test __getattr__ for existing submodules
        assert __getattr__("sub1") == "module_test_package.sub1"
        assert __getattr__("sub2") == "module_test_package.sub2"

        # Test __dir__ returns the list of submodules
        assert __dir__() == submodules

        # Test __all__ contains the submodules
        assert __all__ == submodules


def test_attach_module_raises_attribute_error_for_nonexistent_submodules():
    package_name = "test_package"
    submodules = ["sub1", "sub2"]

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_import_module.side_effect = lambda name: f"module_{name}"

        __getattr__, __dir__, __all__ = attach_module(package_name, submodules)

        # Test __getattr__ raises AttributeError for non-existent submodules
        msg = f"The module '{package_name}' has no attribute 'sub3."
        with pytest.raises(AttributeError, match=msg):
            __getattr__("sub3")
