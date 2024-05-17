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
Exceptions: Miscellaneous
=========================

Custom warnings.
"""

__all__ = ["GeneralWarning", "ParameterWarning", "ToleranceWarning"]


class GeneralWarning(Warning):
    """
    General warning for non-specific issues.
    """


class ParameterWarning(UserWarning):
    """
    Warning for when a parameter is not set.
    """


class ToleranceWarning(UserWarning):
    """
    Warning for unreasonable tolerances.

    If tolerances are too small, the previous step in xitorch's Broyden method
    may become equal to the current step. This leads to a difference of zero,
    which in turn causes `NaN`s due to division by the difference.
    """
