"""
Default settings for `dxtb` calculations.
"""

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


# SCF settings

GUESS = "eeq"
"""Initial guess for orbital charges."""

GUESS_CHOICES = ["eeq", "sad"]
"""List of possible choices for `GUESS`."""

MAXITER = 20
"""Maximum number of SCF iterations."""

VERBOSITY = 1
"""Verbosity of printout."""

XITORCH_VERBOSITY = False
"""Verbosity of printout."""

XITORCH_XATOL = 1.0e-6
"""The absolute tolerance of the norm of the input of the equilibrium function."""

XITORCH_FATOL = 1.0e-6
"""The absolute tolerance of the norm of the output of the equilibrium function."""

# Fermi smearing

ETEMP = 300.0
"""Electronic temperature for Fermi smearing."""

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
