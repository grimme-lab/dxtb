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

import pytest
from unittest.mock import patch
from dxtb._src.loader.lazy import attach_var, attach_vars

import pytest
from unittest.mock import patch, MagicMock


def test_attach_var_imports_variables():
    package_name = "test_package"
    varnames = ["var1", "var2"]

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_module = MagicMock()
        mock_module.var1 = "value1"
        mock_module.var2 = "value2"
        mock_import_module.return_value = mock_module

        __getattr__, __dir__, __all__ = attach_var(package_name, varnames)

        # Test __getattr__ for existing variables
        assert __getattr__("var1") == "value1"
        assert __getattr__("var2") == "value2"

        # Test __dir__ returns the list of variables
        assert __dir__() == varnames

        # Test __all__ contains the variables
        assert __all__ == varnames


def test_attach_var_raises_attribute_error_for_nonexistent_variables():
    package_name = "test_package"
    varnames = ["var1", "var2"]

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_module = MagicMock()
        mock_module.var1 = "value1"
        mock_import_module.return_value = mock_module

        __getattr__, __dir__, __all__ = attach_var(package_name, varnames)

        # Test __getattr__ raises AttributeError for non-existent variables
        msg = f"The module '{package_name}' has no attribute 'var3."
        with pytest.raises(AttributeError, match=msg):
            __getattr__("var3")


def test_attach_vars_imports_variables():
    module_vars = {"package1": ["var1", "var2"], "package2": ["var3", "var4"]}

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_package1 = MagicMock()
        mock_package1.var1 = "value1"
        mock_package1.var2 = "value2"
        mock_package2 = MagicMock()
        mock_package2.var3 = "value3"
        mock_package2.var4 = "value4"
        mock_import_module.side_effect = lambda name: (
            mock_package1 if name == "package1" else mock_package2
        )

        __getattr__, __dir__, __all__ = attach_vars(module_vars)

        # Test __getattr__ for existing variables
        assert __getattr__("var1") == "value1"
        assert __getattr__("var2") == "value2"
        assert __getattr__("var3") == "value3"
        assert __getattr__("var4") == "value4"

        # Test __dir__ returns the list of variables
        assert __dir__() == ["var1", "var2", "var3", "var4"]

        # Test __all__ contains the variables
        assert __all__ == ["var1", "var2", "var3", "var4"]


def test_attach_vars_raises_attribute_error_for_nonexistent_variables():
    module_vars = {"package1": ["var1", "var2"], "package2": ["var3", "var4"]}

    # Mock importlib.import_module to simulate module imports
    with patch("importlib.import_module") as mock_import_module:
        mock_package1 = MagicMock()
        mock_package1.var1 = "value1"
        mock_package1.var2 = "value2"
        mock_package2 = MagicMock()
        mock_package2.var3 = "value3"
        mock_package2.var4 = "value4"
        mock_import_module.side_effect = lambda name: (
            mock_package1 if name == "package1" else mock_package2
        )

        __getattr__, __dir__, __all__ = attach_vars(module_vars)

        # Test __getattr__ raises AttributeError for non-existent variables
        msg = f"No module contains the variable 'var5'."
        with pytest.raises(AttributeError, match=msg):
            __getattr__("var5")
