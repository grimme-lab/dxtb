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
Loaders: Lazy
=============

Loaders and functions for lazy loading of modules and variables.

Example
-------
The following example demonstrates how to use the func:`.attach_module`
function to lazily load submodules of a package.

.. code-block:: python

    from dxtb._src.loader.lazy import attach_module
    __getattr__, __dir__, __all__ = attach_module(__name__, ["sub1", "sub2"])

To improve this setup with type checking to assist with tools like mypy, use
the following pattern:

.. code-block:: python

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from . import sub1 as sub1
        from . import sub2 as sub2
    else:
        import dxtb._src.loader.lazy as _lazy

        __getattr__, __dir__, __all__ = _lazy.attach_module(
            __name__,
            ["sub1", "sub2"],
        )

        del _lazy

In a similar manner, the :func:`.attach_var` function can be used to lazily
load individual variables. Below is an example of how to lazily load a
variable, such as a class:

.. code-block:: python

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from dxtb.mol.molecule import Mol
    else:
        import dxtb._src.loader.lazy as _lazy

        __getattr__, __dir__, __all__ = _lazy.attach_var(
            dxtb.mol.molecule, ["Mol"]
        )

        del _lazy
"""
from .lazy_module import *
from .lazy_param import *
from .lazy_var import *
