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
Parametrization: Tensor Type
============================

Tensor type that allows serialization in Pydantic models.
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from dxtb.typing import Tensor

__all__ = ["TensorPydantic"]


def _tensor_serializer(t: Tensor) -> list[float | int] | float | int:
    return t.tolist() if t.ndim > 0 else t.item()


class TensorPydantic(Tensor):
    """Patched tensor type for serialization with Pydantic."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda v: v,  # identity validator,
            schema=core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _tensor_serializer
            ),
        )
