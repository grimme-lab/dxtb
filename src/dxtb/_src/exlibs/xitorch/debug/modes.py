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
from contextlib import contextmanager

__all__ = [
    "is_debug_enabled",
    "set_debug_mode",
    "enable_debug",
    "disable_debug",
]


class DebugSingleton:
    class __DebugSingleton:
        def __init__(self):
            self._isdebug = False  # default mode is not in the debug mode

        def set_debug_mode(self, mode):
            self._isdebug = mode

        def get_debug_mode(self):
            return self._isdebug

    instance = None

    def __init__(self):
        if DebugSingleton.instance is None:
            DebugSingleton.instance = DebugSingleton.__DebugSingleton()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, val):
        return setattr(self.instance, name, val)


def set_debug_mode(mode):
    dbg_obj = DebugSingleton()
    dbg_obj.set_debug_mode(mode)


def is_debug_enabled():
    dbg_obj = DebugSingleton()
    return dbg_obj.get_debug_mode()


@contextmanager
def enable_debug():
    try:
        dbg_mode = is_debug_enabled()
        set_debug_mode(True)
        yield
    except Exception as e:
        raise e
    finally:
        set_debug_mode(dbg_mode)


@contextmanager
def disable_debug():
    try:
        dbg_mode = is_debug_enabled()
        set_debug_mode(False)
        yield
    except Exception as e:
        raise e
    finally:
        set_debug_mode(dbg_mode)
