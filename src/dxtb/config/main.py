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
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from dxtb.typing import Any, PathLike, Self, get_default_device, get_default_dtype

from ..constants import defaults, labels
from .integral import ConfigIntegrals
from .scf import ConfigSCF


class Config:
    """
    Configuration of the calculation.
    """

    def __init__(
        self,
        *,
        file=None,
        exclude: str | list[str] = defaults.EXCLUDE,
        method: str | int = defaults.METHOD,
        grad: bool = False,
        use_cache: bool = False,
        batch_mode: int = defaults.BATCH_MODE,
        # integrals
        int_cutoff: float = defaults.INTCUTOFF,
        int_driver: str | int = defaults.INTDRIVER,
        int_level: int = defaults.INTLEVEL,
        int_uplo: str = defaults.INTUPLO,
        # PyTorch
        anomaly: bool = False,
        device: torch.device = get_default_device(),
        dtype: torch.dtype = get_default_dtype(),
        # SCF
        maxiter: int = defaults.MAXITER,
        mixer: str = defaults.MIXER,
        damp: float = defaults.DAMP,
        guess: str | int = defaults.GUESS,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        x_atol: float = defaults.X_ATOL,
        f_atol: float = defaults.F_ATOL,
        force_convergence: bool = False,
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: dict = defaults.FERMI_THRESH,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
    ) -> None:
        self.file = file
        self.exclude = exclude
        self.grad = grad
        self.use_cache = use_cache

        self.device = device
        self.dtype = dtype
        self.anomaly = anomaly

        # use property to also set the batch mode in SCF config
        self._batch_mode = batch_mode

        if isinstance(method, str):
            if method.casefold() in labels.GFN1_XTB_STRS:
                self.method = labels.GFN1_XTB
            elif method.casefold() in labels.GFN2_XTB_STRS:
                self.method = labels.GFN2_XTB
            else:
                raise ValueError(f"Unknown xtb method '{method}'.")
        elif isinstance(method, int):
            if method not in (labels.GFN1_XTB, labels.GFN2_XTB):
                raise ValueError(f"Unknown xtb method '{method}'.")

            self.method = method
        else:
            raise TypeError(
                "The method must be of type 'int' or 'str', but "
                f"'{type(method)}' was given."
            )

        self.ints = ConfigIntegrals(
            level=int_level,
            cutoff=int_cutoff,
            driver=int_driver,
            uplo=int_uplo,
        )

        self.scf = ConfigSCF(
            guess=guess,
            maxiter=maxiter,
            mixer=mixer,
            damp=damp,
            scf_mode=scf_mode,
            scp_mode=scp_mode,
            x_atol=x_atol,
            f_atol=f_atol,
            force_convergence=force_convergence,
            batch_mode=batch_mode,
            # SCF: Fermi
            fermi_etemp=fermi_etemp,
            fermi_maxiter=fermi_maxiter,
            fermi_thresh=fermi_thresh,
            fermi_partition=fermi_partition,
            # SCF: PyTorch
            device=device,
            dtype=dtype,
        )

        # compatibility checks
        if (
            self.method == labels.GFN2_XTB
            and self.ints.driver != labels.INTDRIVER_LIBCINT
        ):
            raise RuntimeError(
                "Multipole integrals not available in PyTorch integral drivers."
                " Use `libcint` as backend."
            )

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        return cls(
            # general
            file=args.file,
            exclude=args.exclude,
            method=args.method,
            grad=args.grad,
            use_cache=args.use_cache,
            # integrals
            int_cutoff=args.int_cutoff,
            int_driver=args.int_driver,
            int_level=args.int_level,
            int_uplo=args.int_uplo,
            # PyTorch
            anomaly=args.detect_anomaly,
            device=args.device,
            dtype=args.dtype,
            # SCF
            maxiter=args.maxiter,
            mixer=args.mixer,
            damp=args.damp,
            guess=args.guess,
            scf_mode=args.scf_mode,
            scp_mode=args.scp_mode,
            x_atol=args.xtol,
            f_atol=args.ftol,
            force_convergence=args.force_convergence,
            # SCF: Fermi
            fermi_etemp=args.fermi_etemp,
            fermi_maxiter=args.fermi_maxiter,
            fermi_thresh=args.fermi_thresh,
            fermi_partition=args.fermi_partition,
        )

    @classmethod
    def from_json(cls, path: PathLike) -> Self:
        """
        Create a configuration from a JSON file.

        Parameters
        ----------
        path : PathLike
            The path to the JSON file.

        Returns
        -------
        Self
            The configuration object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File '{path}' does not exist.")

        # pylint: disable=import-outside-toplevel
        import json

        with open(path, encoding="utf-8") as json_file:
            cfg = json.loads(json_file.read())

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """
        Create a configuration from a dictionary.

        Parameters
        ----------
        cfg : dict[str, Any]
            The configuration dictionary.

        Returns
        -------
        Self
            The configuration object.
        """
        # TODO: More sophisticated validation
        return cls(**cfg)

    @property
    def batch_mode(self) -> int:
        return self._batch_mode

    @batch_mode.setter
    def batch_mode(self, value: int) -> None:
        self._batch_mode = value
        self.scf.batch_mode = value

    def info(self) -> dict[str, dict[str, Any]]:
        """
        Return a dictionary with the configuration information.

        Returns
        -------
        dict
            The configuration information.
        """
        return {
            "Calculation Configuration": {
                "Program Call": " ".join(sys.argv),
                "Input File(s)": self.file,
                "Method": labels.GFN_XTB_MAP[self.method],
                "Excluded": False if len(self.exclude) == 0 else self.exclude,
                "Gradient": self.grad,
                "Integral driver": labels.INTDRIVER_MAP[self.ints.driver],
                "FP accuracy": str(self.dtype),
                "Device": str(self.device),
            },
            **self.scf.info(),
        }

    def to_json(self, path: PathLike | None = None) -> str:
        """
        Serialize the configuration to a JSON-formatted string.

        Returns:
            str: A JSON-formatted string representing the configuration.
        """
        # pylint: disable=import-outside-toplevel
        import json

        config_info = self.info()

        def serialize(value):
            if isinstance(value, torch.device) or isinstance(value, torch.dtype):
                return str(value)
            elif isinstance(value, list):
                # Recursively serialize lists
                return [serialize(v) for v in value]
            elif isinstance(value, dict):
                # Recursively serialize dicts
                return {k: serialize(v) for k, v in value.items()}
            else:
                return value

        # Serialize the entire configuration info to JSON
        serialized_info = {k: serialize(v) for k, v in config_info.items()}

        # Convert the dictionary to a JSON string
        json_string = json.dumps(serialized_info, indent=4)

        if path is not None:
            path = Path(path)
            if path.exists():
                path.unlink()

            with open(path, "w", encoding="utf-8") as json_file:
                json_file.write(json_string)

        return json_string

    def __str__(self) -> str:
        info = self.info()["SCF Options"]
        info_str = ", ".join(f"{key}={value}" for key, value in info.items())
        return f"{self.__class__.__name__}({info_str})"

    def __repr__(self) -> str:
        return str(self)
