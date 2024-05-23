# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
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
import inspect

from dxtb._src.exlibs.xitorch._core.editable_module import EditableModule

__all__ = [
    "assert_broadcastable",
    "assert_fcn_params",
    "assert_runtime",
    "assert_type",
]


def assert_broadcastable(shape1, shape2):
    if len(shape1) > len(shape2):
        assert_broadcastable(shape2, shape1)
        return
    for a, b in zip(shape1[::-1], shape2[::-1][: len(shape1)]):
        assert (
            a == 1 or b == 1 or a == b
        ), f"The shape {shape1} and {shape2} are not broadcastable"


def assert_fcn_params(fcn, args):
    if inspect.ismethod(fcn) and isinstance(fcn.__self__, EditableModule):
        fcn.__self__.assertparams(fcn, *args)


def assert_runtime(cond, msg=""):
    if not cond:
        raise RuntimeError(msg)


def assert_type(cond, msg=""):
    if not cond:
        raise TypeError(msg)
