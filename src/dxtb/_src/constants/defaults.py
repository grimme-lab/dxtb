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
Default Settings
================

This module contains the defaults for all `dxtb` calculations.
"""

from __future__ import annotations

import torch

# General

STRICT = False
"""
Strict mode. Always throws errors if ``True``, instead of making sensible adaptations.
"""

BATCH_MODE = 0
"""Batch mode for calculation."""

BATCH_MODE_CHOICES = [0, 1, 2]
"""
List of possible choices for `BATCH_MODE`:

- 0: No batching
- 1: Regular batching with padding
- 2: Batched calculation without padding
"""

STEP_SIZE = 1e-5
"""Step size for numerical differentiation."""

EINSUM_OPTIMIZE = "greedy"
"""Optimization algorithm for `einsum`."""

METHOD = "gfn1"
"""General method for calculation from the xtb family."""

METHOD_CHOICES = ["gfn1", "gfn1-xtb", "gfn2", "gfn2-xtb"]
"""List of possible choices for `METHOD`."""

SPIN = None
"""Number of unpaired electrons of the system."""

CHRG = 0
"""Total charge of the system."""

EXCLUDE: list[str] = []
"""List of xTB components to exclude during the calculation."""

EXCLUDE_CHOICES = ["disp", "rep", "hal", "es2", "es3", "scf", "all"]
"""List of possible choices for `EXCLUDE`."""

MAX_ELEMENT = 86
"""Maximum atomic number for the calculation."""

# Integral settings

INTCUTOFF = 50.0
"""Real-space cutoff (in Bohr) for integral evaluation. (50.0)"""

INTDRIVER = "libcint"
"""Integral driver."""

INTDRIVER_CHOICES = [
    "dxtb",
    "dxtb2",
    "torch",
    "torch2",
    "pytorch",
    "pytorch2",
    "libcint",
    "c",
]
"""List of possible choices for `INTDRIVER`."""

INTLEVEL = 2
"""Determines types of calculated integrals."""

INTLEVEL_CHOICES = [0, 1, 2, 3, 4, 5]
"""List of possible choices for `INTLEVEL`."""

INTUPLO = "l"
"""Integral mode for PyTorch integral calculation."""

INTUPLO_CHOICES = ["n", "N", "l", "L", "u", "U"]
"""List of possible choices for `INTUPLO`."""


DP_SHAPE = 3
"""Number of dimensions of the dipole integral."""

QP_SHAPE = 6
"""
Number of dimension of the quadrupole integral. Libcint returns 9, which can be
reduced to 6 due to symmetry (tracless representation).
"""


# SCF settings

GUESS = "eeq"
"""Initial guess for orbital charges."""

GUESS_CHOICES = ["eeq", "sad"]
"""List of possible choices for `GUESS`."""

DAMP = 0.3
"""Damping factor for mixing in SCF iterations."""

MAXITER = 100
"""Maximum number of SCF iterations."""

MIXER = "broyden"
"""SCF mixing scheme for convergence acceleration."""

MIXER_CHOICES = ["anderson", "broyden", "simple"]
"""List of possible choices for ``MIXER``."""

SCF_MODE = "nonpure"
"""
Whether to use full gradient tracking in SCF, make use of the implicit
function theorem as provided by ``xitorch.optimize.equilibrium``, or use the
experimental single-shot procedure.
"""

SCF_MODE_CHOICES = [
    "default",
    "implicit",
    "nonpure",
    "full",
    "full_tracking",
    "full-tracking",
    "experimental",
    "single_shot",
    "single-shot",
]
"""List of possible choices for ``SCF_MODE``."""

SCP_MODE = "potential"
"""
Type of self-consistent parameter, i.e., which quantity is converged in the SCF
iterations.
"""

SCP_MODE_CHOICES = ["charge", "charges", "potential", "fock"]
"""
List of possible choices for ``SCP_MODE``. 'charge' and 'charges' are identical.
"""

SCF_FORCE_CONVERGENCE = False
"""Whether to continue with un-converged results."""

VERBOSITY = 5
"""Verbosity of printout."""

LOG_LEVEL = "info"
"""Default logging level."""

LOG_LEVEL_CHOICES = ["critical", "error", "warn", "warning", "info", "debug"]
"""List of possible choices for `LOG_LEVEL`."""

XITORCH_VERBOSITY = False
"""Verbosity of printout."""

X_ATOL = 1.0e-4
"""
The absolute tolerance of the norm of the input of the equilibrium function.
"""

F_ATOL = 1.0e-4
"""
The absolute tolerance of the norm of the output of the equilibrium function.
"""

# Fermi smearing

FERMI_ETEMP = 300.0
"""Electronic temperature for Fermi smearing in K."""

FERMI_MAXITER = 200
"""Maximum number of iterations for Fermi smearing."""

FERMI_THRESH = {
    torch.float16: torch.tensor(1e-2, dtype=torch.float16),
    torch.float32: torch.tensor(1e-5, dtype=torch.float32),
    torch.float64: torch.tensor(1e-10, dtype=torch.float64),
}
"""Convergence thresholds for different float data types."""

FERMI_PARTITION = "equal"
"""Partitioning scheme for electronic free energy."""

FERMI_PARTITION_CHOICES = ["equal", "atomic"]
"""List of possible choices for `FERMI_PARTITION`."""

# cache

CACHE_ENABLED = False
"""Whether caching is enabled."""

CACHE_STORE_HCORE = False
"""Whether to store the core Hamiltonian."""

CACHE_STORE_OVERLAP = False
"""Whether to store the overlap integral matrix."""

CACHE_STORE_DIPOLE = False
"""Whether to store the dipole integral matrix."""

CACHE_STORE_QUADRUPOLE = False
"""Whether to store the quadrupole integral matrix."""

CACHE_STORE_CHARGES = True
"""Whether to store the atomic charges."""

CACHE_STORE_COEFFICIENTS = False
"""Whether to store the MO coefficients."""

CACHE_STORE_DENSITY = False
"""Whether to store the density matrix."""

CACHE_STORE_FOCK = False
"""Whether to store the Fock matrix."""

CACHE_STORE_ITERATIONS = True
"""Whether to store the number of SCF iterations."""

CACHE_STORE_MO_ENERGIES = False
"""Whether to store the MO energies."""

CACHE_STORE_OCCUPATIONS = False
"""Whether to store the orbital occupation."""

CACHE_STORE_POTENTIAL = False
"""Whether to store the potential."""

# PyTorch

TORCH_DTYPE = torch.double
"""Default data type for floating point tensors."""

TORCH_DTYPE_CHOICES = ["float16", "float32", "float64", "double", "sp", "dp"]
"""List of possible choices for `TORCH_DTYPE`."""

TORCH_DEVICE = "cpu"
"""Default device for tensors."""

TORCH_DEVICE_CHOICES = ["cpu", "cuda"]
"""List of possible choices for `TORCH_DEVICE`."""

PAD = 0
"""Defaults value to indicate padding."""

PADNZ = -9999999
"""Default non-zero value to indicate padding."""
