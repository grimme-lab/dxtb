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
External Libraries
==================

All external libraries used by `dxtb` are imported here. They are lazily loaded
to reduce import times and to avoid unnecessary imports if the library is not
used.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import libcint as libcint
    from . import pyscf as pyscf
    from . import scipy as scipy
    from . import xitorch as xitorch
else:
    import dxtb._src.loader.lazy as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        ["libcint", "pyscf", "scipy", "xitorch"],
    )

    del _lazy

del TYPE_CHECKING
