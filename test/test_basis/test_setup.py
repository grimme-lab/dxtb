"""
Test setup of the basis set.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import Basis, IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.utils import batch

from .samples import samples

sample_list = ["H2", "LiH", "Li2", "H2O", "S", "SiH4", "MB16_43_01"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    atombasis = bas.create_dqc(positions)
    assert isinstance(atombasis, list)
    assert not isinstance(atombasis[0], list)

    for basis in atombasis:
        assert not isinstance(basis, list)

        if basis.atomz == 1:
            assert [b.angmom for b in basis.bases] == [0, 0]
        elif basis.atomz in (3, 4, 5, 6, 7, 8, 9, 11):
            assert [b.angmom for b in basis.bases] == [0, 1]
        elif basis.atomz in (13, 14, 16, 17):
            assert [b.angmom for b in basis.bases] == [0, 1, 2]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    pos_dict = {
        0: sample1["positions"].to(**dd),
        1: sample2["positions"].to(**dd),
    }
    positions = batch.pack((pos_dict[0], pos_dict[1]), value=float("nan"))

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    atombasis_batch = bas.create_dqc(positions)
    assert isinstance(atombasis_batch, list)
    assert isinstance(atombasis_batch[0], list)

    # test by re-assembling the positions
    for i, atombasis in enumerate(atombasis_batch):
        pos = []
        assert isinstance(atombasis, list)

        for basis in atombasis:
            pos.append(basis.pos)

            if basis.atomz == 1:
                assert [b.angmom for b in basis.bases] == [0, 0]
            elif basis.atomz in (3, 4, 5, 6, 7, 8, 9, 11):
                assert [b.angmom for b in basis.bases] == [0, 1]
            elif basis.atomz in (13, 14, 16, 17):
                assert [b.angmom for b in basis.bases] == [0, 1, 2]

        assert pytest.approx(pos_dict[i]) == torch.stack(pos)
