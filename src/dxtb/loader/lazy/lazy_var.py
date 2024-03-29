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
A LazyLoader class for loading variables.
"""

from __future__ import annotations

import importlib

from tad_mctc.typing import Any, Callable, Sequence


def attach_var(package_name: str, varnames: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
    __all__: list[str] = list(varnames)

    def __getattr__(name: str) -> Any:
        if name not in varnames:
            raise AttributeError(
                f"The module '{package_name}' has no attribute '{name}."
            )

        module = importlib.import_module(f"{package_name}")

        return getattr(module, name)

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__
