"""
Default Settings
================

This module contains the defaults for all `dxtb` calculations.
"""
from __future__ import annotations

import torch

# General

METHOD = "gfn1"
"""General method for calculation from the xtb family."""

METHOD_CHOICES = ["gfn1", "gfn1-xtb", "gfn2", "gfn2-xtb"]
"""List of possible choices for `METHOD`."""

SPIN = None
"""Total spin of the system."""

CHRG = None
"""Total charge of the system."""

THRESH = {
    torch.float16: torch.tensor(1e-2, dtype=torch.float16),
    torch.float32: torch.tensor(1e-5, dtype=torch.float32),
    torch.float64: torch.tensor(1e-10, dtype=torch.float64),
}
"""Convergence thresholds for different float data types."""

EXCLUDE: list[str] = []
"""List of xTB components to exclude during the calculation."""

EXCLUDE_CHOICES = ["disp", "rep", "hal", "es2", "es3", "scf", "all"]
"""List of possible choices for `EXCLUDE`."""

# Integral settings

INTCUTOFF = 50.0
"""Real-space cutoff (in Angstrom) for integral evaluation. (50.0)"""

INTDRIVER = "libcint"
"""Integral driver."""

INTLEVEL = 1
"""Determines types of calculated integrals."""

DP_SHAPE = 3
"""Number of dimensions of the dipole integral."""

QP_SHAPE = 9
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

MAXITER = 20
"""Maximum number of SCF iterations."""

MIXER = "broyden"
"""SCF mixing scheme for convergence acceleration."""

MIXER_CHOICES = ["anderson", "broyden", "simple"]
"""List of possible choices for `MIXER`."""

SCF_MODE = "default"
"""
Whether to use full gradient tracking in SCF, make use of the implicit
function theorem as provided by `xitorch.optimize.equilibrium`, or use the
experimental single-shot procedure.
"""

SCF_MODE_CHOICES = ["default", "implicit", "full", "full_tracking", "experimental"]
"""List of possible choices for `SCF_MODE`."""

SCP_MODE = "potential"
"""
Type of self-consistent parameter, i.e., which quantity is converged in the SCF
iterations.
"""

SCP_MODE_CHOICES = ["charge", "charges", "potential", "fock"]
"""
List of possible choices for `SCP_MODE`. 'charge' and 'charges' are identical.
"""

VERBOSITY = 1
"""Verbosity of printout."""

XITORCH_VERBOSITY = False
"""Verbosity of printout."""

XITORCH_XATOL = 1.0e-5
"""
The absolute tolerance of the norm of the input of the equilibrium function.
"""

XITORCH_FATOL = 1.0e-5
"""
The absolute tolerance of the norm of the output of the equilibrium function.
"""

# Fermi smearing

ETEMP = 300.0
"""Electronic temperature for Fermi smearing in K."""

FERMI_MAXITER = 200
"""Maximum number of iterations for Fermi smearing."""

FERMI_FENERGY_PARTITION = "equal"
"""Partitioning scheme for electronic free energy."""

FERMI_FENERGY_PARTITION_CHOICES = ["equal", "atomic"]
"""List of possible choices for `FERMI_FENERGY_PARTITION`."""

# PyTorch

TORCH_DTYPE = torch.float32
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
