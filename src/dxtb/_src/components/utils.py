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
Components: Utility
===================

Utility functions for components documentation.
"""

__all__ = [
    "_docstring_reset",
    "_docstring_update",
]


def _docstring_update(func):
    """
    Decorator to assign a generic docstring to update methods.
    The docstring is generated based on the method name.
    """
    attribute_name = func.__name__.replace("update_", "")

    docstring = f"""
    Update the attribute of the '{attribute_name}' object based on the provided
    arguments in the keyword arguments.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments containing the attributes and their new values to
        be updated in the component object.

    Returns
    -------
    Component
        The component object with the updated attributes.

    Raises
    ------
    ValueError
        If no component with the given label is found in the list.

    Examples
    --------
    See `ComponentList.update`.
    """
    func.__doc__ = docstring
    return func


def _docstring_reset(func):
    """
    Decorator to assign a generic docstring to update methods.
    The docstring is generated based on the method name.
    """
    attribute_name = func.__name__.replace("update_", "")

    docstring = f"""
    Reset the attributes of the '{attribute_name}' object within the list.

    This method resets any tensor attributes to a detached clone of their
    original state. The `requires_grad` status of each tensor is preserved.

    Returns
    -------
    Component
        The component object with the resetted attributes.

    Raises
    ------
    ValueError
        If no component with the given label is found in the list.
    """
    func.__doc__ = docstring
    return func
