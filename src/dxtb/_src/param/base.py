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
Parametrization: Base
=====================

Definition of the full parametrization data for the extended tight-binding
methods.

The dataclass can represent a complete parametrization file produced by the
`tblite`_ library, however it only stores the raw data rather than the full
representation, i.e., the transformation to the corresponding atom-resolved
quantities must be carried out separately.

.. _tblite: https://tblite.readthedocs.io
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from dxtb._src.typing import Any, PathLike, Self, Type

from .charge import Charge
from .dispersion import Dispersion
from .element import Element
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .multipole import Multipole
from .repulsion import Repulsion
from .thirdorder import ThirdOrder

__all__ = ["Param"]


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.

    The parametrization of a calculator with the model data must account for
    missing transformations, like extracting the principal quantum numbers from
    the shells. The respective checks are therefore deferred to the
    instantiation of the calculator, while a deserialized model in `tblite`_ is
    already verified at this stage.

    .. _tblite: https://tblite.readthedocs.io
    """

    meta: Optional[Meta] = None
    """Descriptive data on the model."""

    element: Dict[str, Element]
    """Element specific parameter records."""

    hamiltonian: Optional[Hamiltonian] = None
    """Definition of the Hamiltonian, always required."""

    dispersion: Optional[Dispersion] = None
    """Definition of the dispersion correction."""

    repulsion: Optional[Repulsion] = None
    """Definition of the repulsion contribution."""

    charge: Optional[Charge] = None
    """Definition of the isotropic second-order charge interactions."""

    multipole: Optional[Multipole] = None
    """Definition of the anisotropic second-order multipolar interactions."""

    halogen: Optional[Halogen] = None
    """Definition of the halogen bonding correction."""

    thirdorder: Optional[ThirdOrder] = None
    """Definition of the isotropic third-order charge interactions."""

    def clean_model_dump(self) -> dict[str, Any]:
        """
        Clean the model from any `None` values.
        """

        return self.model_dump(exclude_none=True)

    @classmethod
    def from_file(cls: Type[Self], filepath: PathLike) -> Self:
        """
        Load a parametrization from a file. The file format is determined by the
        file extension. Supported formats are JSON, TOML, and YAML.

        Parameters
        ----------
        filepath : PathLike
            The file path to the parametrization file.

        Returns
        -------
        Param
            The loaded parametrization data.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            return cls.from_json_file(filepath)
        if filepath.suffix == ".toml":
            return cls.from_toml_file(filepath)
        if filepath.suffix in (".yaml", ".yml"):
            return cls.from_yaml_file(filepath)

        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def to_file(self, filepath: PathLike, **kwargs) -> None:
        """
        Save the parametrization to a file. The file format is determined by the
        file extension. Supported formats are JSON, TOML, and YAML.

        Parameters
        ----------
        filepath : PathLike
            The file path to save the parametrization data.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            self.to_json_file(filepath, **kwargs)
        elif filepath.suffix == ".toml":
            self.to_toml_file(filepath, **kwargs)
        elif filepath.suffix in (".yaml", ".yml"):
            self.to_yaml_file(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    @classmethod
    def from_json_file(cls: Type[Self], filepath: PathLike) -> Self:
        import json

        with open(filepath, encoding="utf-8") as fd:
            return cls(**json.load(fd))

    def to_json_file(self, filepath: PathLike, **kwargs) -> None:
        import json

        with open(filepath, "w", encoding="utf-8") as fd:
            json.dump(self.clean_model_dump(), fd, **kwargs)

    @classmethod
    def from_toml_file(cls: Type[Self], filepath: PathLike) -> Self:
        try:
            import tomli as toml
        except ImportError:
            raise ImportError(
                "A TOML package is required for TOML support. "
                "You can install it via `pip install tomli`."
            )

        with open(filepath, "rb") as fd:
            return cls(**toml.load(fd))

    def to_toml_file(self, filepath: PathLike, **kwargs) -> None:
        try:
            import tomli_w as toml_w  # type: ignore
        except ImportError:
            try:
                import toml as toml_w  # type: ignore
            except ImportError:
                raise ImportError(
                    "A TOML writer package is required for TOML support. "
                    "You can install one via `pip install tomli-w`."
                )

        with open(filepath, "wb") as fd:
            toml_w.dump(self.clean_model_dump(), fd, **kwargs)  # type: ignore

    @classmethod
    def from_yaml_file(cls: Type[Self], filepath: Path) -> Self:
        """
        Load a parametrization from a YAML file.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "The PyYAML package is required for YAML support. "
                "You can install it via `pip install pyyaml`."
            )

        with open(filepath, encoding="utf-8") as fd:
            return cls(**yaml.safe_load(fd))

    def to_yaml_file(self, filepath: Path, **kwargs) -> None:
        """
        Save the parametrization to a YAML file.

        Parameters
        ----------
        filepath : Path
            The file path to save the parametrization data.

        Raises
        ------
        ImportError
            If the PyYAML package is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "The PyYAML package is required for YAML support. "
                "You can install it via `pip install pyyaml`."
            )

        with open(filepath, "w", encoding="utf-8") as fd:
            yaml.dump(self.clean_model_dump(), fd, encoding="utf-8", **kwargs)
