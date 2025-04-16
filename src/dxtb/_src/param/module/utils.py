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
Parametrization: Differentiable Parametrization
===============================================

This module provides a differentiable representation of the
extended tight-binding parametrization using PyTorch.

The :class:`.ParamModule` class automatically converts a Pydantic
model into a hierarchical :class:`~torch.nn.Module` tree.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from tad_mctc.data import pse
from torch import nn

from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.typing import DD, Any, Tensor
from dxtb._src.utils import is_int_list

from .types import NonNumericValue, ParameterModule

__all__ = ["ParamElementsPairsMixin"]


class ParamGetterMixin:
    """Collection of methods to access parameters for unique elements."""

    device: torch.device | None
    """Device of the tensors."""

    dtype: torch.dtype
    """Data type of the tensors."""

    dd: DD
    """Dictionary with device and dtype information."""

    parameter_tree: nn.Module
    """The parameter tree containing all parameters."""

    def get(self, *keys: str | int, unwrapped: bool = True) -> Any:
        """
        Access nested parameters via keys. If the final node is a
        :class:`ParameterModule`, returns its underlying
        :class:`~torch.nn.Parameter`.

        Parameters
        ----------
        *keys : sequence of str or int
            The keys (or indices) to traverse the parameter tree.
            Keys can be provided in a single string (e.g.
            ``par.get("element.H")``) or as a sequence of strings
            (e.g. ``par.get("element", "H")``).
        unwrapped : bool, default True
            If ``True``, unwraps :class:`ParameterModule` to return the
            underlying tensor.

        Returns
        -------
        Any
            The retrieved parameter or value.

        Raises
        ------
        KeyError
            If any key is not found.
        """
        if len(keys) == 0:
            raise KeyError("No keys provided.")

        if len(keys) == 1:
            key = keys[0]
            if isinstance(key, str):
                if key != "element" and "." in key:
                    keys = tuple(key.split("."))

        node: Any = self.parameter_tree
        for k in keys:
            if isinstance(node, nn.ModuleDict):
                if not isinstance(k, str):
                    raise TypeError("Keys for ModuleDict must be of type str")
                node = node[k]
                continue
            if isinstance(node, nn.ModuleList):
                if not isinstance(k, int):
                    raise TypeError("Keys for ModuleList must be of type int")
                node = node[k]
                continue
            if isinstance(node, NonNumericValue):
                node = node.value
                continue
            raise KeyError(f"Key '{k}' not found in parameter tree.")

        # Unwrap if the final node is still wrapped
        if isinstance(node, NonNumericValue):
            return node.value

        if unwrapped and isinstance(node, ParameterModule):
            return node.param
        return node

    # Recursively set differentiable (requires_grad=True) for the parameter(s)

    def set_differentiable(
        self, *keys: str | int, ignore_non_numeric: bool = True
    ) -> None:
        """
        Recursively set the parameter(s) at the given key path to be
        differentiable (i.e. set their :attr:`requires_grad` attribute to
        ``True``).
        If the key path corresponds to a container (e.g. a branch such as
        "charge"), all numeric leaves underneath will be updated.

        Parameters
        ----------
        *keys : sequence of str or int
            Keys (or indices) to traverse the parameter tree to the target node.
        ignore_non_numeric : bool, default True
            Whether to ignore non-numeric (nonnumeric) fields encountered in the
            branch.

        Raises
        ------
        KeyError
            If the key path is not found.
        TypeError
            If a non-numeric (non-numeric) field is encountered and
            ``ignore_non_numeric`` is False.
        """
        try:
            node = self.get(*keys, unwrapped=False)
        except KeyError as e:
            raise KeyError(
                f"Key path {'->'.join(map(str, keys))} not found."
            ) from e

        if isinstance(node, NonNumericValue) and not ignore_non_numeric:
            raise TypeError(
                "Cannot set a nonnumeric value to be differentiable."
            )
        self._set_differentiable_recursive(node, ignore_non_numeric)

    def _set_differentiable_recursive(
        self, module: nn.Module, ignore_non_numeric: bool
    ) -> None:
        """
        Recursively sets all :class:`ParameterModule` instances within *module*
        (or *module* itself if it is a :class:`ParameterModule`) to be
        differentiable.

        Numeric leaves will have their :attr:`requires_grad` attribute set to
        ``True``. If a non-numeric value is encountered and
        ``ignore_non_numeric`` is ``True``, it is skipped; otherwise a
        :class:`TypeError` is raised.

        Parameters
        ----------
        module : nn.Module
            The module to update.
        ignore_non_numeric : bool
            Whether to ignore non-numeric values during the update.

        Raises
        ------
        TypeError
            If a non-numeric (nonnumeric) value is encountered and
            ``ignore_non_numeric`` is ``False``.
        """
        if isinstance(module, ParameterModule):
            module.param.requires_grad_(True)
            return

        if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            for child in module.children():
                self._set_differentiable_recursive(child, ignore_non_numeric)
            return

        if isinstance(module, NonNumericValue):
            if ignore_non_numeric is True:
                return
            raise TypeError(
                "Cannot set a non-numeric value to be differentiable."
            )

        raise TypeError(
            f"Unsupported type {type(module)} encountered in "
            "`set_differentiable`. Cannot set to differentiable."
        )

    # Shortcuts for checking existence

    def __contains__(self, key: str) -> bool:
        """
        Returns True if the top-level key exists in the parameter tree.

        Only applicable if the underlying tree is a
        :class:`~torch.nn.ModuleDict`.
        """
        if isinstance(self.parameter_tree, nn.ModuleDict):
            return key in self.parameter_tree
        return False

    def is_none(self, *keys: str | int) -> bool:
        """
        Check whether branch specified by *keys* exists and its unwrapped
        value is ``None``.

        Returns ``True`` if branch is missing or its value (when unwrapped)
        equals ``None``.
        """
        try:
            value = self.get(*keys, unwrapped=True)
            return value is None
        except KeyError:
            # If the branch does not exist, treat it as None.
            return True

    def is_false(self, *keys: str | int) -> bool:
        """
        Check whether branch specified by *keys* exists and its unwrapped
        value is exactly ``False``.

        Returns ``True`` if branch exists and its value equals ``False``,
        else ``False``.
        """
        try:
            value = self.get(*keys, unwrapped=True)
            return value is False
        except KeyError:
            return False


class ParamShortcutMixin(ParamGetterMixin):
    """Shortcut methods to access the parameter tree."""

    @property
    def meta(self) -> SimpleNamespace:
        """
        Return the metadata of the parametrization as an object with attribute access.

        If the meta data is stored as a dictionary, it is converted into a SimpleNamespace,
        so that one can do, for example, `if self.meta.name: ...`.
        """
        return SimpleNamespace(**self.get("meta", unwrapped=True))

    @property
    def element(self) -> nn.ModuleDict:
        """
        Convenience property to access the element-specific branch as an
        :class:`~torch.nn.ModuleDict`.
        """
        elem = self.get("element", unwrapped=False)
        if not isinstance(elem, nn.ModuleDict):
            raise TypeError("The 'element' branch is not a ModuleDict.")
        return elem


class ParamElementsPairsMixin(ParamShortcutMixin):
    """Collection of methods to access parameters for unique elements."""

    device: torch.device | None
    """Device of the tensors."""

    dtype: torch.dtype
    """Data type of the tensors."""

    dd: DD
    """Dictionary with device and dtype information."""

    parameter_tree: nn.Module
    """The parameter tree containing all parameters."""

    def get_elem_param(
        self,
        unique: Tensor,
        key: str,
        pad_val: int = -1,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """
        Obtain a element-wise parametrized quantity for selected atomic numbers.

        Parameters
        ----------
        unique : Tensor
            Unique atomic numbers in the system (shape: ``(nunique,)``).
        key : str
            Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).
        pad_val : int, optional
            Value to pad the tensor with. Default is `-1`.
        dtype : torch.dtype | None
            Data type of the tensor. If ``None`` (default), the data type of
            the class is used.

        Returns
        -------
        Tensor
            Parametrization of selected elements.

        Raises
        ------
        ValueError
            If the type of the value of `key` is neither `float` nor `int`.
        """
        _dtype = dtype if dtype is not None else self.dtype

        collected = []

        def tensor1d(val: int | float) -> torch.Tensor:
            return torch.tensor([val], device=self.device, dtype=_dtype)

        for number in torch.atleast_1d(unique):
            symbol = pse.Z2S.get(int(number), "X")

            # If symbol does not exist, it will be treated as padding
            if symbol not in self.element:
                collected.append(tensor1d(pad_val))
                continue

            p = self.element[symbol]
            assert isinstance(p, nn.ModuleDict)

            if key not in p:
                raise KeyError(
                    f"The key '{key}' is not in the element parameterization"
                )

            param_module = p[key]

            if not isinstance(param_module, ParameterModule):
                raise TypeError(
                    f"Expected ParameterModule, got {type(param_module)}"
                )

            # Convert scalar values to 1D tensors for concatenation
            collected.append(torch.atleast_1d(param_module.param))

        if len(collected) > 0:
            return torch.cat(collected, dim=0)

        return torch.tensor([], device=self.device, dtype=_dtype)

    def get_elem_angular(self) -> dict[int, list[int]]:
        """
        Obtain angular momenta of the shells of all atoms.

        Returns
        -------
        dict[int, list[int]]
            Angular momenta of all elements.
        """
        label2angular = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

        result: dict[int, list[int]] = {}

        for sym in self.element.keys():
            num = pse.S2Z[sym]

            # Actually, `shells` is a ModuleDict, but to satisfy the
            # type checker for the following listcomp, we use NonNumericValue.
            shells: list[NonNumericValue] = self.get("element", sym, "shells")

            # The `shells` is list of strings, like ["1s", "2p", ...].
            result[num] = [label2angular[label[-1]] for label in shells]

        return result

    def get_pair_param(self, symbols: list[str] | list[int]) -> Tensor:
        """
        Obtain tensor of a pair-wise parametrized quantity for all unique pairs.

        Parameters
        ----------
        symbols : list[str | int]
            List of atomic symbols or atomic numbers.
        par_pair : dict[str, float]
            Parametrization of pairs.

        Returns
        -------
        Tensor
            Parametrization of all pairs of ``symbols``.
        """
        par_pair = self.get("hamiltonian.xtb.kpair", unwrapped=False)
        if not isinstance(par_pair, nn.ModuleDict):
            raise TypeError("The 'kpair' branch is not a ModuleDict.")

        if is_int_list(symbols):
            symbols = [pse.Z2S.get(i, "X") for i in symbols]

        ndim = len(symbols)
        pair_mat = torch.ones((ndim, ndim), **self.dd)
        for i, isp in enumerate(symbols):
            for j, jsp in enumerate(symbols):
                # Watch format! ("element1-element2")
                key1 = f"{isp}-{jsp}"
                key2 = f"{jsp}-{isp}"

                if key1 in par_pair:
                    p = par_pair[key1]
                    assert isinstance(p, ParameterModule)
                    value = p.param.view(-1)[0]
                elif key2 in par_pair:
                    p = par_pair[key2]
                    assert isinstance(p, ParameterModule)
                    value = p.param.view(-1)[0]
                else:
                    value = torch.tensor(1.0, **self.dd)

                pair_mat[i, j] = value

        return pair_mat

    def get_elem_valence(self, unique: Tensor, pad_val: int = -1) -> Tensor:
        """
        Obtain valence of the shells of all atoms.

        Parameters
        ----------
        unique : Tensor
            Unique elements in the system (shape: ``(nunique,)``).
        pad_val : int, optional
            Value to pad the tensor with. Default is -1.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each element.
        """
        vals_list = []
        key = "shells"
        label2angular = {
            "s": 0,
            "p": 1,
            "d": 2,
            "f": 3,
            "g": 4,
        }

        par = self.element

        for number in torch.atleast_1d(unique):
            # Resolve element symbol; using pse.Z2S as a mapping.
            el = pse.Z2S.get(int(number.item()), "X")
            shells = []
            if el in par:
                # Iterate over shells stored in the element parameters.
                for shell in getattr(par[el], key):
                    shell_type = shell[-1]
                    if shell_type not in label2angular:
                        raise ValueError(f"Unknown shell type '{shell_type}'.")
                    shells.append(label2angular[shell_type])
            else:
                shells = [pad_val]

            # Create tensor of shell labels
            r = torch.tensor(shells, dtype=torch.long, device=self.device)
            tmp = torch.ones(r.shape, dtype=torch.bool, device=self.device)

            if r.size(0) < 2:
                vals = tmp
            else:
                # Sort so duplicate values appear together
                y, idxs = torch.sort(r, stable=True)
                # Mark unique values (duplicates become False)
                tmp[1:] = (y[1:] - y[:-1]) != 0
                # Recover the original ordering
                _, idxs = torch.sort(idxs)
                vals = torch.gather(tmp, 0, idxs)

            for val in vals:
                vals_list.append(val)

        return torch.stack(vals_list)

    def get_elem_pqn(
        self,
        unique: Tensor,
        pad_val: int = -1,
    ) -> Tensor:
        """
        Obtain principal quantum numbers of the shells of all atoms.

        Parameters
        ----------
        unique : Tensor
            Unique elements in the system (shape: ``(nunique,)``).
        pad_val : int, optional
            Value to pad the tensor with. Default is `-1`.

        Returns
        -------
        Tensor
            Principal quantum numbers of the shells of all atoms. Always of
            type :data:`dxtb._src.constants.defaults.DEFAULT_BASIS_INT`.
        """
        key = "shells"
        shells = []
        par = self.element

        for number in torch.atleast_1d(unique):
            el = pse.Z2S.get(int(number.item()), "X")
            if el in par:
                for shell in getattr(par[el], key):
                    shells.append(int(shell[0]))
            else:
                shells.append(pad_val)

        return torch.tensor(shells, device=self.device, dtype=DEFAULT_BASIS_INT)
