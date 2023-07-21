"""
Integral container
==================

A class that acts as a container for integrals.
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike

__all__ = ["Integrals"]


class Integrals(TensorLike):
    """
    Integral container.
    """

    __slots__ = ["_hcore", "_overlap", "_dipole", "_quad", "_run_checks"]

    def __init__(
        self,
        hcore: Tensor | None = None,
        overlap: Tensor | None = None,
        dipole: Tensor | None = None,
        quad: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        self._hcore = hcore
        self._overlap = overlap
        self._dipole = dipole
        self._quad = quad
        self._run_checks = True

    @property
    def hcore(self) -> Tensor:
        if self._hcore is None:
            raise ValueError("No Core Hamiltonian provided.")
        return self._hcore

    @hcore.setter
    def hcore(self, hcore: Tensor) -> None:
        self._hcore = hcore
        self.checks()

    @property
    def overlap(self) -> Tensor:
        if self._overlap is None:
            raise ValueError("No overlap integral provided.")
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: Tensor) -> None:
        self._overlap = overlap
        self.checks()

    @property
    def dipole(self) -> Tensor | None:
        """
        Dipole integral of shape (3, nao, nao).

        Returns
        -------
        Tensor | None
            Dipole integral if set, else `None`.
        """
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: Tensor) -> None:
        self._dipole = dipole
        self.checks()

    @property
    def quad(self) -> Tensor | None:
        return self._quad

    @quad.setter
    def quad(self, quad: Tensor) -> None:
        self._quad = quad
        self.checks()

    # checks

    @property
    def run_checks(self) -> bool:
        return self._run_checks

    @run_checks.setter
    def run_checks(self, run_checks: bool) -> None:
        current = self.run_checks
        self._run_checks = run_checks

        # switching from False to True should automatically run checks
        if current is False and run_checks is True:
            self.checks()

    def checks(self) -> None:
        """
        Checks the shapes of the tensors.

        Expected shapes:
        - hcore and overlap: (batch_size, nao, nao) or (nao, nao)
        - dipole: (batch_size, 3, nao, nao) or (3, nao, nao)
        - quad: (batch_size, 6, nao, nao) or (6, nao, nao)

        Raises
        ------
        ValueError:
            If any of the tensors have incorrect shapes or inconsistent batch
            sizes.
        """
        if self.run_checks is False:
            return

        nao = None
        batch_size = None

        for name in ["hcore", "overlap", "dipole", "quad"]:
            tensor = getattr(self, "_" + name)
            if tensor is None:
                continue

            if name in ["hcore", "overlap"]:
                if len(tensor.shape) not in [2, 3]:
                    raise ValueError(
                        f"Tensor '{name}' must have 2 or 3 dimensions. "
                        f"Got {len(tensor.shape)}."
                    )
                if len(tensor.shape) == 3:
                    if batch_size is not None and tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"Tensor '{name}' has a different batch size. "
                            f"Expected {batch_size}, got {tensor.shape[0]}."
                        )
                    batch_size = tensor.shape[0]
                nao = tensor.shape[-1]
            else:  # dipole or quad
                if len(tensor.shape) not in [3, 4]:
                    raise ValueError(
                        f"Tensor '{name}' must have 3 or 4 dimensions. "
                        f"Got {len(tensor.shape)}."
                    )
                if len(tensor.shape) == 4:
                    if batch_size is not None and tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"Tensor '{name}' has a different batch size. "
                            f"Expected {batch_size}, got {tensor.shape[0]}."
                        )
                    batch_size = tensor.shape[0]
                nao = tensor.shape[-2]

            if tensor.shape[-2:] != (nao, nao):
                raise ValueError(
                    f"Tensor '{name}' last two dimensions should be "
                    f"(nao, nao). Got {tensor.shape[-2:]}."
                )
            if name == "dipole" and tensor.shape[-3] != 3:
                raise ValueError(
                    f"Tensor '{name}' third to last dimension should be 3. "
                    f"Got {tensor.shape[-3]}."
                )
            if name == "quad" and tensor.shape[-3] != 6:
                raise ValueError(
                    f"Tensor '{name}' third to last dimension should be 6. "
                    f"Got {tensor.shape[-3]}."
                )
