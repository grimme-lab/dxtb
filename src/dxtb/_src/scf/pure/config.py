"""
SCF Configuration
=================

This module defines a storage class for the SCF options.
"""

from __future__ import annotations

import torch
from tad_mctc.units.energy import KELVIN2AU

from dxtb._src.constants import defaults
from dxtb._src.typing import Any

from .data import _Data

__all__ = ["SCFConfig"]


class SCFConfig:
    """
    Self-consistent field configuration, as pure base class containing only
    configuration information.

    This class should _not_ contain any tensors, which store AG gradients
    during SCF iterations.
    """

    fwd_options: dict[str, Any]
    """Options for forwards pass"""

    bck_options: dict[str, Any]
    """Options for backwards pass"""

    eigen_options: dict[str, Any]
    """Options for eigensolver"""

    scf_options: dict[str, Any]
    """
    Options for SCF:

    - "etemp": Electronic temperature (in a.u.) for Fermi smearing.
    - "fermi_maxiter": Maximum number of iterations for Fermi smearing.
    - "fermi_thresh": Float data type dependent threshold for Fermi iterations.
    - "fermi_fenergy_partition": Partitioning scheme for electronic free energy.
    """

    use_potential: bool
    """Whether to use the potential or the charges"""

    batch_mode: int
    """Whether multiple systems or a single one are handled"""

    def __init__(self, data: _Data, batch_mode: int, **kwargs: Any) -> None:
        self.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}
        self.fwd_options = {
            "force_convergence": False,
            "method": "broyden1",
            "alpha": -0.5,
            "f_tol": defaults.F_ATOL,
            "x_tol": defaults.X_ATOL,
            "f_rtol": float("inf"),
            "x_rtol": float("inf"),
            "maxiter": defaults.MAXITER,
            "verbose": False,
            "line_search": False,
            **kwargs.pop("fwd_options", {}),
        }

        self.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

        self.scf_options = {**kwargs.pop("scf_options", {})}
        self.scp_mode = self.scf_options.get("scp_mode", defaults.SCP_MODE)

        # Only infer shapes and types from _Data (no logic involved),
        # i.e. keep _Data and SCFConfig instances disjunct objects.
        self._shape = data.ints.hcore.shape
        self._dtype = data.ints.hcore.dtype
        self._device = data.ints.hcore.device

        self.kt = data.ints.hcore.new_tensor(
            self.scf_options.get("etemp", defaults.FERMI_ETEMP) * KELVIN2AU
        )
        self.batch_mode = batch_mode

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the density matrix in this engine.
        """
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        return self._dtype

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        return self._device
