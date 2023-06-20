"""
Test culling (removing of systems) from IndexHelper.
"""
from __future__ import annotations

import torch

from dxtb._types import Slicers, Tensor
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch


def test_culling() -> None:
    numbers = batch.pack(
        [
            torch.tensor([3, 1]),  # LiH
            torch.tensor([14, 1, 1, 1, 1]),  # SiH4
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    ref_ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    conv = torch.tensor([False, True])
    slicers: Slicers = {
        "orbital": [slice(0, i) for i in [torch.tensor(6)]],
        "shell": [slice(0, i) for i in [torch.tensor(4)]],
        "atom": [slice(0, i) for i in [torch.tensor(2)]],
    }

    ihelp.cull(conv, slicers=slicers)

    for name in ihelp.__slots__:
        attr = getattr(ihelp, name)
        if isinstance(attr, Tensor):
            if name not in ("unique_angular", "ushells_to_unique"):
                # get attribute from normal ihelp and remove padding
                a = getattr(ref_ihelp, name)[0]
                ref = batch.deflate(a, value=a[-1])

                assert (ref == attr.squeeze()).all()


def test_no_action() -> None:
    numbers = batch.pack(
        [
            torch.tensor([3, 1]),  # LiH
            torch.tensor([14, 1, 1, 1, 1]),  # SiH4
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    ref_ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    conv = torch.tensor([False, True])
    slicers: Slicers = {
        "orbital": (...,),
        "shell": (...,),
        "atom": (...,),
    }

    ihelp.cull(conv, slicers=slicers)

    for name in ihelp.__slots__:
        attr = getattr(ihelp, name)
        if isinstance(attr, Tensor):
            if name not in ("unique_angular", "ushells_to_unique"):
                ref = getattr(ref_ihelp, name)[0]
                assert (ref == attr.squeeze()).all()
