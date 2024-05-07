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
Components: Base Classes
========================

Base classes for all tight-binding components and component lists and caches.
"""

from dxtb._src.components.base import Component as Component
from dxtb._src.components.base import ComponentCache as ComponentCache
from dxtb._src.components.classicals.base import Classical as Classical
from dxtb._src.components.classicals.base import ClassicalABC as ClassicalABC
from dxtb._src.components.classicals.base import ClassicalCache as ClassicalCache
from dxtb._src.components.interactions.base import Interaction as Interaction
from dxtb._src.components.interactions.base import InteractionCache as InteractionCache
from dxtb._src.components.interactions.list import InteractionList as InteractionList
from dxtb._src.components.interactions.list import (
    InteractionListCache as InteractionListCache,
)

__all__ = [
    "Component",
    "ComponentCache",
    #
    "Classical",
    "ClassicalABC",
    "ClassicalCache",
    #
    "Interaction",
    "InteractionList",
    "InteractionListCache",
]
