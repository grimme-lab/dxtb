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
Labels: SCF
===========

Labels for SCF-related options.
"""

# guess
GUESS_EEQ = 0
"""Integer code for EEQ guess."""
GUESS_EEQ_STRS = ("eeq", "equilibration")
"""String codes for EEQ guess."""

GUESS_SAD = 1
"""Integer code for SAD (superposition of atomic density; zero) guess."""
GUESS_SAD_STRS = ("sad", "zero")
"""String codes for SAD (superposition of atomic density; zero) guess."""

GUESS_MAP = ["EEQ", "SAD"]
"""String map (for printing) of SCF guess methods."""

# backward differentiation method
SCF_MODE_FULL = 0
"""Integer code for SCF with full tracking (unrolling) for Autograd."""

SCF_MODE_FULL_STRS = ("full", "full_tracking", "unrolling")
"""String codes for SCF with full tracking (unrolling) for Autograd."""

SCF_MODE_IMPLICIT = 1
"""Integer code for SCF using implicit function theorem for differentiation."""

SCF_MODE_IMPLICIT_STRS = ("default", "implicit")
"""String codes for SCF using implicit function theorem for differentiation."""

SCF_MODE_IMPLICIT_NON_PURE = 2
"""Integer code for non-pure version of implicitly differentiated SCF."""

SCF_MODE_IMPLICIT_NON_PURE_STRS = (
    "implicit_old",
    "implicit_nonpure",
    "nonpure",
    "non-pure",
    "old",
)
"""String codes for non-pure version of implicitly differentiated SCF."""

SCF_MODE_EXPERIMENTAL = 3
"""Integer code for SCF with single-shot gradient."""

SCF_MODE_EXPERIMENTAL_STRS = (
    "experimental",
    "perfect",
    "single-shot",
    "single_shot",
)
"""String codes for SCF with single-shot gradient."""

SCF_MODE_MAP = [
    "Full Tracking (unrolling)",
    "implicit",
    "implicit (non-pure/old)",
    "experimental",
]
"""String map (for printing) of SCF modes."""

# convergence target
SCP_MODE_FOCK = 0
"""Integer code for SCF convergence targets."""

SCP_MODE_FOCK_STRS = ("fock", "fockian")
"""String codes for SCF convergence targets."""

SCP_MODE_CHARGE = 1
"""Integer code for SCF convergence targets."""

SCP_MODE_CHARGE_STRS = ("charge", "charges")
"""String codes for SCF convergence targets."""

SCP_MODE_POTENTIAL = 2
"""Integer code for SCF convergence targets."""

SCP_MODE_POTENTIAL_STRS = ("potential", "pot")
"""String codes for SCF convergence targets."""

SCP_MODE_MAP = ["Fock matrix", "charges", "potential"]
"""String map (for printing) of SCF convergence targets."""

# fermi partition
FERMI_PARTITION_EQUAL = 0
"""Integer code for equal Fermi partition."""

FERMI_PARTITION_EQUAL_STRS = ("equal", "same")
"""String codes for equal Fermi partition."""

FERMI_PARTITION_ATOMIC = 1
"""Integer code for atomic Fermi partition."""

FERMI_PARTITION_ATOMIC_STRS = ("atom", "atomic")
"""String codes for atomic Fermi partition."""

FERMI_PARTITION_MAP = ["equal", "atom"]
"""String map (for printing) of Fermi partitioning methods."""

# mixer
MIXER_LINEAR = 0
"""Integer code for linear/simple mixing."""

MIXER_LINEAR_STRS = ("linear", "l", "simple", "s")
"""String codes for linear/simple mixing."""

MIXER_ANDERSON = 1
"""Integer code for Anderson mixing."""

MIXER_ANDERSON_STRS = ("anderson", "a")
"""String codes for Anderson mixing."""

MIXER_BROYDEN = 2
"""Integer code for Broyden mixing."""

MIXER_BROYDEN_STRS = ("broyden", "b")
"""String codes for Broyden mixing."""

MIXER_MAP = ["Linear", "Anderson", "Broyden"]
"""String map (for printing) of mixing methods."""
