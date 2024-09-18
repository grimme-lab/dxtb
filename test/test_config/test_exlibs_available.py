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
Test availability of external libraries.
"""

from dxtb._src.exlibs.available import has_libcint, has_pyscf, has_scipy


def test_libcint() -> None:
    if has_libcint is True:
        assert has_libcint is True
    else:
        assert has_libcint is False


def test_pyscf() -> None:
    if has_pyscf is True:
        assert has_pyscf is True
    else:
        assert has_pyscf is False


def test_scipy() -> None:
    if has_scipy is True:
        assert has_scipy is True
    else:
        assert has_scipy is False
