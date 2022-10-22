"""
Default settings for `dxtb` calculations.
"""

import torch

THRESH = {
    torch.float16: torch.tensor(1e-2, dtype=torch.float16),
    torch.float32: torch.tensor(1e-5, dtype=torch.float32),
    torch.float64: torch.tensor(1e-10, dtype=torch.float64),
}
"""Convergence thresholds for different float data types."""

# SCF settings

GUESS = "eeq"  # "sad"
"""Initial guess for orbital charges."""

MAXITER = 20
"""Maximum number of SCF iterations."""

VERBOSITY = 1
"""Verbosity of printout."""

# Fermi smearing

ETEMP = 300
"""Electronic temperature for Fermi smearing."""

FERMI_MAXITER = 200
"""Maximum number of iterations for Fermi smearing."""

FERMI_FENERGY_PARTITION = "equal"  # "atomic"
"""Partitioning scheme for electronic free energy."""
