"""
Test for fractional occupation (Fermi smearing).
Reference values obtained with tbmalt.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.constants import K2AU
from dxtb.param import GFN1_XTB, get_elem_angular
from dxtb.scf.iterator import SelfConsistentField
from dxtb.utils import batch
from dxtb.wavefunction import filling

from .samples import samples

sample_list = ["H", "H2", "LiH", "SiH4", "S2"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail(dtype: torch.dtype):
    dd: DD = {"device": device, "dtype": dtype}

    evals = torch.arange(1, 6, dtype=dtype)
    nel = torch.tensor([4.0, 4.0], dtype=dtype)

    #
    with pytest.raises(RuntimeError):
        filling.get_alpha_beta_occupation(nel, uhf=torch.tensor([0.0]))

    # wrong type
    with pytest.raises(TypeError):
        kt = 300.0
        filling.get_fermi_occupation(nel, evals, kt)  # type: ignore

    # negative etemp
    with pytest.raises(ValueError):
        kt = torch.tensor(-1.0, dtype=dtype)
        filling.get_fermi_occupation(nel, evals, kt)

    # no electrons
    with pytest.raises(ValueError):
        _nel = torch.tensor(0.0, dtype=dtype)
        filling.get_fermi_occupation(_nel, evals, None)

    # convergence fails
    with pytest.raises(RuntimeError):
        sample = samples["SiH4"]
        emo = sample["emo"].to(**dd)
        nel = sample["n_electrons"].to(**dd)
        nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))
        kt = torch.tensor(10000 * K2AU, dtype=dtype)
        filling.get_fermi_occupation(nab, emo, kt, maxiter=1)


@pytest.mark.parametrize("uhf", [[0, 0, 0], [1, 1, 0], [3, 1, 0]])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_uhf(dtype: torch.dtype, uhf: list):
    dd: DD = {"device": device, "dtype": dtype}

    with pytest.raises(ValueError):
        nel = torch.tensor([2, 1, 2], **dd)
        filling.get_alpha_beta_occupation(nel, nel.new_tensor(uhf))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]

    nel = sample["n_electrons"].to(**dd)
    uhf = sample["spin"].to(**dd)
    nab = filling.get_alpha_beta_occupation(nel, uhf)

    emo = sample["emo"].to(**dd)

    ref_focc = sample["focc"].to(**dd)
    ref_efermi = sample["e_fermi"].to(**dd)

    kt = emo.new_tensor(300 * K2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi, abs=tol) == efermi

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc, abs=tol) == focc


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]

    nel = batch.pack(
        [
            sample1["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
        ]
    )
    uhf = batch.pack(
        [
            sample1["spin"].to(**dd),
            sample2["spin"].to(**dd),
            sample2["spin"].to(**dd),
        ]
    )
    nab = filling.get_alpha_beta_occupation(nel, uhf)

    emo = batch.pack(
        [
            sample1["emo"].to(**dd),
            sample2["emo"].to(**dd),
            sample2["emo"].to(**dd),
        ]
    )
    # emo = emo.unsqueeze(-2).expand([*nab.shape, -1]) # if only one ref channel

    ref_efermi = batch.pack(
        [
            sample1["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
        ]
    )
    # ref_efermi = ref_efermi.expand([*nab.shape]) # if only one ref channel

    ref_focc = batch.pack(
        [
            sample1["focc"].to(**dd),
            sample2["focc"].to(**dd),
            sample2["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * K2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi, abs=tol) == efermi

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc, abs=tol) == focc


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("kt", [0.0, 5000.0])
def test_kt(dtype: torch.dtype, kt: float):
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["SiH4"]
    numbers = sample["numbers"].to(device)

    nel = sample["n_electrons"].to(**dd)
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = sample["emo"].to(**dd)

    ref_fenergy = {
        0.0: emo.new_tensor(0.0),
        5000.0: emo.new_tensor(-1.6758385176418445e-004),
    }
    ref_focc = {
        0.0: emo.new_tensor(
            [
                2.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        5000.0: emo.new_tensor(
            [
                1.9999999835050197,
                1.9998302497800045,
                1.9998302497800045,
                1.9998302497800045,
                1.6888179157220813e-004,
                1.6888179157220572e-004,
                1.6888179157220273e-004,
                9.3970980586158547e-007,
                9.3970980586158028e-007,
                2.2294515200028562e-007,
                2.2294515200028403e-007,
                2.2294515200027612e-007,
                7.3530914097230426e-008,
                4.7590701921523143e-020,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
            ]
        ),
    }

    # occupation
    focc = filling.get_fermi_occupation(nab, emo, emo.new_tensor(kt * K2AU))
    assert torch.allclose(ref_focc[kt], focc.sum(-2), atol=tol)

    # electronic free energy
    d = torch.zeros_like(focc)  # dummy
    scf = SelfConsistentField(
        d, d, d, focc, d, numbers, d, d, scf_options={"etemp": kt}  # type: ignore
    )
    fenergy = scf.get_electronic_free_energy()
    assert pytest.approx(ref_fenergy[kt], abs=tol, rel=tol) == fenergy.sum(-1)

    scf.scf_options["fermi_fenergy_partition"] = "wrong"
    with pytest.raises(ValueError):
        scf.get_electronic_free_energy()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_lumo_not_existing(dtype: torch.dtype) -> None:
    """Helium has no LUMO due to the minimal basis."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples["He"]

    nel = batch.pack(
        [
            sample["n_electrons"].to(**dd),
            sample["n_electrons"].to(**dd),
            sample["n_electrons"].to(**dd),
        ]
    )
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = batch.pack(
        [
            sample["emo"].to(**dd),
            sample["emo"].to(**dd),
            sample["emo"].to(**dd),
        ]
    )

    ref_efermi = batch.pack(
        [
            sample["e_fermi"].to(**dd),
            sample["e_fermi"].to(**dd),
            sample["e_fermi"].to(**dd),
        ]
    )
    ref_focc = batch.pack(
        [
            sample["focc"].to(**dd),
            sample["focc"].to(**dd),
            sample["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * K2AU)

    efermi, _ = filling.get_fermi_energy(nab, emo)
    assert pytest.approx(ref_efermi, abs=tol) == efermi

    focc = filling.get_fermi_occupation(nab, emo, kt)
    assert pytest.approx(ref_focc, abs=tol) == focc


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_lumo_obscured_by_padding(dtype: torch.dtype) -> None:
    """A missing LUMO can be obscured by padding."""
    dd: DD = {"device": device, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples["H2"], samples["He"]

    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    nel = batch.pack(
        [
            sample1["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
        ]
    )
    nab = filling.get_alpha_beta_occupation(nel, torch.zeros_like(nel))

    emo = batch.pack(
        [
            sample1["emo"].to(**dd),
            sample2["emo"].to(**dd),
            sample2["emo"].to(**dd),
        ]
    )

    ref_efermi = batch.pack(
        [
            sample1["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
            sample2["e_fermi"].to(**dd),
        ]
    )
    ref_focc = batch.pack(
        [
            sample1["focc"].to(**dd),
            sample2["focc"].to(**dd),
            sample2["focc"].to(**dd),
        ]
    )

    kt = emo.new_tensor(300 * K2AU)

    mask = ihelp.orbitals_per_shell
    mask = mask.unsqueeze(-2).expand([*nab.shape, -1])

    efermi, _ = filling.get_fermi_energy(nab, emo, mask=mask)
    assert pytest.approx(ref_efermi, abs=tol) == efermi

    focc = filling.get_fermi_occupation(nab, emo, kt, mask=mask)
    assert pytest.approx(ref_focc, abs=tol) == focc
