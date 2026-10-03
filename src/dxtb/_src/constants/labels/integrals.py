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
Labels: Integrals
=================

All labels related to integrals and their computation.
"""

__all__ = [
    "INTDRIVER_LIBCINT",
    "INTDRIVER_LIBCINT_STRS",
    "INTDRIVER_PYTORCH",
    "INTDRIVER_PYTORCH_STRS",
    "INTDRIVER_MAP",
    "INTDRIVER_AUTOGRAD",
    "INTDRIVER_AUTOGRAD_STRS",
    "INTALGORITHM_CHOICES",
    "INTALGORITHM_DEFAULT",
    #
    "INTLEVEL_NONE",
    "INTLEVEL_OVERLAP",
    "INTLEVEL_HCORE",
    "INTLEVEL_DIPOLE",
    "INTLEVEL_QUADRUPOLE",
    "INTLEVEL_MAX",
]

# integral drivers

INTDRIVER_LIBCINT = 0
"""Integer code for LIBCINT driver."""

INTDRIVER_LIBCINT_STRS = ("libcint", "c")
"""String codes for LIBCINT driver."""

INTDRIVER_PYTORCH = 1
"""Integer code for the PyTorch driver."""

INTDRIVER_PYTORCH_STRS = ("autograd", "pytorch", "torch", "dxtb")
"""String codes for the PyTorch driver."""

INTDRIVER_AUTOGRAD = INTDRIVER_PYTORCH
"""Deprecated alias of :data:`INTDRIVER_PYTORCH`."""

INTDRIVER_AUTOGRAD_STRS = INTDRIVER_PYTORCH_STRS
"""Deprecated alias of :data:`INTDRIVER_PYTORCH_STRS`."""

INTDRIVER_MAP = {
    INTDRIVER_LIBCINT: "libcint",
    INTDRIVER_PYTORCH: "PyTorch",
}
"""String map (for printing) of integral drivers."""

# algorithms of the PyTorch driver

INTALGORITHM_CHOICES = ("md", "os")
"""
Names of the 1D kernels of the PyTorch driver: McMurchie-Davidson with
Hermite moments (``md``) and the three-index Obara-Saika recursion (``os``).
"""

INTALGORITHM_DEFAULT = "os"
"""Kernel of the PyTorch driver used when none is requested."""

# levels

INTLEVEL_NONE = 0
"""No integrals."""

INTLEVEL_OVERLAP = 1
"""Overlap integrals."""

INTLEVEL_HCORE = 2
"""Core Hamiltonian integrals."""

INTLEVEL_DIPOLE = 3
"""Dipole integrals."""

INTLEVEL_QUADRUPOLE = 4
"""Quadrupole integrals."""

INTLEVEL_MAX = INTLEVEL_QUADRUPOLE
"""Maximum integral level."""
