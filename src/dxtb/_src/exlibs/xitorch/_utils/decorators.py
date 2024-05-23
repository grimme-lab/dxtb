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
import functools
import inspect
import warnings

__all__ = ["deprecated"]


def deprecated(date_str):
    return lambda obj: _deprecated(obj, date_str)


def _deprecated(obj, date_str):
    if inspect.isfunction(obj):
        name = "Function %s" % (obj.__str__())
    elif inspect.isclass(obj):
        name = "Class %s" % (obj.__name__)

    if inspect.ismethod(obj) or inspect.isfunction(obj):

        @functools.wraps(obj)
        def fcn(*args, **kwargs):
            warnings.warn(f"{name} is deprecated since {date_str}", stacklevel=2)
            return obj(*args, **kwargs)

        return fcn

    elif inspect.isclass(obj):
        # replace the __init__ function
        old_init = obj.__init__

        @functools.wraps(old_init)
        def newinit(*args, **kwargs):
            warnings.warn(f"{name} is deprecated since {date_str}", stacklevel=2)
            return old_init(*args, **kwargs)

        obj.__init__ = newinit
        return obj
