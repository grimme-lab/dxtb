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
Integral Namemanager
====================

The libcint interface is accessed via strings. This module provides the
corresponding name handling and manipulation.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict

from dxtb.typing import Sequence

from .symmetry import s1


class IntorNameManager:
    """
    Class for integral name manipulation.

    This class should only perform string-manipulation and no array operations.
    """

    # ops name must not contain sep name
    ops_name = ["ip", "rr"]  # name of basis operators
    sep_name = ["a", "b"]  # separator of basis (other than the middle operator)

    # components shape of raw operator and basis operators
    # should be a tuple with AT MOST 1 element
    rawop_comp = defaultdict(
        tuple,
        {  # type: ignore
            "r0": (3,),
            "r0r0": (9,),
            "r0r0r0": (27,),
            "j": (3,),
            "jj": (9,),
            "jjj": (27,),
            "m": (3,),
            "mm": (9,),
            "mmm": (27,),
            "n": (3,),
            "nn": (9,),
            "nnn": (27,),
        },
    )
    op_comp = defaultdict(
        tuple,
        {  # type: ignore
            "ip": (3,),
        },
    )

    # the number of new dimensions added with the operators
    rawop_ndim = defaultdict(int, {k: len(v) for (k, v) in rawop_comp.items()})
    op_ndim = defaultdict(int, {k: len(v) for (k, v) in op_comp.items()})

    def __init__(self, int_type: str, shortname: str):
        self._int_type = int_type
        self._shortname = shortname
        self._rawop, self._ops = self.split_name(int_type, shortname)
        self._nbasis = len(self._ops)

        # middle index (where the rawops should be)
        self._imid = (self._nbasis + 1) // 2

    @property
    def fullname(self):
        return self._int_type + "_" + self._shortname

    @property
    def rawopname(self):
        return self._rawop

    @property
    def int_type(self):
        return self._int_type

    @property
    def shortname(self):
        return self._shortname

    @property
    def order(self) -> int:
        """
        Get the order of the derivative of the integral.

        Returns
        -------
        int
            Order of derivative.
        """
        derivative_order = 0
        for ops in self._ops:
            for op in ops:
                if op == "ip":
                    derivative_order += 1
                elif op == "ipip":
                    derivative_order += 2
                elif op == "ipipip":
                    derivative_order += 3

        return derivative_order

    def get_intgl_name(self, spherical: bool) -> str:
        """
        Get the full name of the integral in libcint library.

        Parameters
        ----------
        spherical : bool
            Whether the integral is in spherical or cartesian coordinates.

        Returns
        -------
        str
            Full name of the integral in libcint library.
        """
        cartsph = "sph" if spherical else "cart"
        return f"{self.fullname}_{cartsph}"

    def get_ft_intgl_name(self, spherical: bool) -> str:
        """
        Get the full name of the Fourier transform integral in libcint library.

        Parameters
        ----------
        spherical : bool
            Whether the integral is in spherical or cartesian coordinates.

        Returns
        -------
        str
            Full name of the Fourier transform integral in libcint library.

        Raises
        ------
        NotImplementedError
            If the Fourier transform integral is not implemented for the given
            integral type.
        """
        cartsph = "sph" if spherical is True else "cart"
        int_type = self._int_type
        if int_type == "int1e":
            return f"GTO_ft_{self._shortname}_{cartsph}"

        raise NotImplementedError(f"FT integral for {int_type} not implemented.")

    def get_intgl_deriv_namemgr(self, derivop: str, ibasis: int) -> IntorNameManager:
        """
        Get the name manager of a new integral when derivop is applied to
        ibasis-th basis.

        Parameters
        ----------
        derivop : str
            String of the derivative operation.
        ibasis : int
            Which basis the derivative operation should be performed (0-based).

        Returns
        -------
        IntorNameManager
            Name manager of the new integral.
        """
        assert derivop in self.ops_name
        assert ibasis < self._nbasis

        ops = copy.copy(self._ops)
        ops[ibasis] = [derivop] + ops[ibasis]
        sname = self.join_name(self._int_type, self._rawop, ops)
        return IntorNameManager(self._int_type, sname)

    def get_intgl_deriv_newaxispos(self, derivop: str, ibasis: int) -> None | int:
        """
        Get the new axis position in the new integral name when derivop is applied

        Parameters
        ----------
        derivop : str
            String of the derivative operation.
        ibasis : int
            Which basis the derivative operation should be performed (0-based).

        Returns
        -------
        None | int
            New axis position or None if no new axis is inserted.
        """
        # get how many new axes the operator is going to add
        op_ndim = self.op_ndim[derivop]
        if op_ndim == 0:
            return None

        ops_flat: list[str] = sum(self._ops[:ibasis], [])
        new_ndim = sum(self.op_ndim[op] for op in ops_flat)

        # check if rawsname should also be included
        include_rname = ibasis >= self._imid
        if include_rname:
            new_ndim += self.rawop_ndim[self._rawop]

        return new_ndim

    def get_intgl_components_shape(self) -> tuple[int, ...]:
        # returns the component shape of the array of the given integral
        ops_flat_l: list[str] = sum(self._ops[: self._imid], [])
        ops_flat_r: list[str] = sum(self._ops[self._imid :], [])
        comp_shape = (
            sum([self.op_comp[op] for op in ops_flat_l], ())
            + self.rawop_comp[self._rawop]
            + sum([self.op_comp[op] for op in ops_flat_r], ())
        )
        return comp_shape  # type: ignore

    def get_intgl_symmetry(self, _: Sequence[int]) -> s1.S1Symmetry:
        return s1.S1Symmetry()

    def get_transpose_path_to(
        self, other: IntorNameManager
    ) -> list[tuple[int, int]] | None:
        """
        Get the transpose path to the other integral. Check if the other
        integral can be achieved by transposing the current integral.

        Parameters
        ----------
        other : IntorNameManager
            The other integral name manager.

        Returns
        -------
        list[tuple[int, int]] | None
            Transpose path of `self` to get the same result as the `other`
            integral or `None` if it cannot be achieved.

        Raises
        ------
        RuntimeError
            If the number of basis is not supported.
        """
        nbasis = self._nbasis
        # get the basis transpose paths
        if nbasis == 2:
            transpose_paths: list[list[tuple[int, int]]] = [
                [],
                [(-1, -2)],
            ]
        elif nbasis == 3:
            # NOTE: the third basis is usually an auxiliary basis which
            # typically different from the first two
            transpose_paths = [
                [],
                [(-2, -3)],
            ]
        elif nbasis == 4:
            transpose_paths = [
                [],
                [(-3, -4)],
                [(-1, -2)],
                [(-1, -3), (-2, -4)],
                [(-1, -3), (-2, -4), (-2, -1)],
                [(-1, -3), (-2, -4), (-3, -4)],
            ]
        else:
            raise self._nbasis_error(nbasis)

        def _swap(p: list[list[str]], path: list[tuple[int, int]]) -> list[list[str]]:
            # swap the pattern according to the given transpose path
            r = p[:]  # make a copy
            for i0, i1 in path:
                r[i0], r[i1] = r[i1], r[i0]
            return r

        # try all the transpose path until gets a match
        for transpose_path in transpose_paths:
            # pylint: disable=protected-access
            if _swap(self._ops, transpose_path) == other._ops:
                return transpose_path
        return None

    def get_comp_permute_path(self, transpose_path: list[tuple[int, int]]) -> list[int]:
        """
        Get the component permute path given the basis transpose path.

        Parameters
        ----------
        transpose_path : list[tuple[int, int]]
            Transpose path of the basis.

        Returns
        -------
        list[int]
            Component permute path.
        """
        # flat_ops: list[str] = sum(self._ops, [])
        # n_ip = flat_ops.count("ip")

        # get the positions of the axes
        dim_pos = []
        ioffset = 0
        for i, ops in enumerate(self._ops):
            if i == self._imid:
                naxes = self.rawop_ndim[self._rawop]
                dim_pos.append(list(range(ioffset, ioffset + naxes)))
                ioffset += naxes
            naxes = sum([self.op_ndim[op] for op in ops])
            dim_pos.append(list(range(ioffset, ioffset + naxes)))
            ioffset += naxes

        # add the bases' axes (assuming each basis only occupy one axes)
        for i in range(self._nbasis):
            dim_pos.append([ioffset])
            ioffset += 1

        # swap the axes
        for t0, t1 in transpose_path:
            s0 = t0 + self._nbasis
            s1 = t1 + self._nbasis
            s0 += 1 if s0 >= self._imid else 0
            s1 += 1 if s1 >= self._imid else 0
            dim_pos[s0], dim_pos[s1] = dim_pos[s1], dim_pos[s0]

        # flatten the list to get the permutation path
        dim_pos_flat: list[int] = sum(dim_pos, [])
        return dim_pos_flat

    @classmethod
    def split_name(cls, int_type: str, shortname: str) -> tuple[str, list[list[str]]]:
        """
        Split the shortname into operator per basis.

        Parameters
        ----------
        int_type : str
            Type of the integral.
        shortname : str
            Shortname of the integral.

        Returns
        -------
        tuple[str, list[list[str]]]
            Raw shortname (i.e., the middle operator) and list of basis-operator shortname.

        Raises
        ------
        RuntimeError
            If the number of basis is not supported.
        """
        deriv_ops = cls.ops_name
        deriv_pattern = re.compile("(" + ("|".join(deriv_ops)) + ")")

        # get the raw shortname (i.e. shortname without derivative operators)
        rawsname = shortname
        for op in deriv_ops:
            rawsname = rawsname.replace(op, "")

        nbasis = cls.get_nbasis(int_type)
        if nbasis == 2:
            ops_str = shortname.split(rawsname)
        elif nbasis == 3:
            assert rawsname.startswith("a"), rawsname
            rawsname = rawsname[1:]
            ops_l, ops_r = shortname.split(rawsname)
            ops_l1, ops_l2 = ops_l.split("a")
            ops_str = [ops_l1, ops_l2, ops_r]
        elif nbasis == 4:
            assert rawsname.startswith("a") and rawsname.endswith("b"), rawsname
            rawsname = rawsname[1:-1]
            ops_l, ops_r = shortname.split(rawsname)
            ops_l1, ops_l2 = ops_l.split("a")
            ops_r1, ops_r2 = ops_r.split("b")
            ops_str = [ops_l1, ops_l2, ops_r1, ops_r2]
        else:
            raise cls._nbasis_error(nbasis)

        ops = [re.findall(deriv_pattern, op_str) for op_str in ops_str]
        assert len(ops) == nbasis
        return rawsname, ops

    @classmethod
    def join_name(cls, int_type: str, rawsname: str, ops: list[list[str]]) -> str:
        """
        Join the raw shortname and list of basis operators into a shortname.

        Parameters
        ----------
        int_type : str
            Type of the integral.
        rawsname : str
            Raw shortname (i.e., the middle operator).
        ops : list[list[str]]
            List of basis-operator shortname.

        Returns
        -------
        str
            Shortname of the integral.

        Raises
        ------
        RuntimeError
            If the number of basis is not supported.
        """
        nbasis = cls.get_nbasis(int_type)
        ops_str = ["".join(op) for op in ops]
        assert len(ops_str) == nbasis

        if nbasis == 2:
            return ops_str[0] + rawsname + ops_str[1]
        elif nbasis == 3:
            return ops_str[0] + cls.sep_name[0] + ops_str[1] + rawsname + ops_str[2]
        elif nbasis == 4:
            return (
                ops_str[0]
                + cls.sep_name[0]
                + ops_str[1]
                + rawsname
                + ops_str[2]
                + cls.sep_name[1]
                + ops_str[3]
            )
        else:
            raise cls._nbasis_error(nbasis)

    @classmethod
    def get_nbasis(cls, int_type: str) -> int:
        """
        Get the number of basis for the given integral type.

        Parameters
        ----------
        int_type : str
            Type of the integral.

        Returns
        -------
        int
            Number of basis.

        Raises
        ------
        RuntimeError
            If the integral type is unknown.
        """
        if int_type in ("int1e", "int2c2e"):
            return 2
        if int_type == "int3c2e":
            return 3
        if int_type == "int2e":
            return 4

        raise RuntimeError(f"Unknown integral type: {int_type}")

    @classmethod
    def _nbasis_error(cls, nbasis: int):
        return RuntimeError(f"Unknown integral with {nbasis} basis")

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            f"(int_type={self._int_type!r}, shortname={self._shortname!r})"
        )

    def __repr__(self) -> str:
        return str(self)
