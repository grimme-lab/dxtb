"""
Test spread and reduce capabilities of IndexHelper.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.basis import IndexHelper
from dxtb.utils import batch, symbol2number


def test_spread() -> None:
    unique = torch.tensor([1, 6])  # include sorting
    atom = torch.tensor([6, 1, 1, 1, 1])
    shell = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1, 1, 1])
    orbital = torch.tensor([6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1])

    angular = {
        1: [0, 0],  # H
        6: [0, 1],  # C
    }
    ihelp = IndexHelper.from_numbers(atom, angular)

    assert (ihelp.spread_uspecies_to_atom(unique) == atom).all()
    assert (ihelp.spread_uspecies_to_shell(unique) == shell).all()
    assert (ihelp.spread_uspecies_to_orbital(unique) == orbital).all()
    assert (ihelp.spread_atom_to_shell(atom) == shell).all()
    assert (ihelp.spread_atom_to_orbital(atom) == orbital).all()
    assert (ihelp.spread_atom_to_shell(atom) == shell).all()
    assert (ihelp.spread_atom_to_orbital(atom) == orbital).all()
    assert (ihelp.spread_shell_to_orbital(shell) == orbital).all()

    # spread a "unique shell"-resolved property
    ushell = ihelp.unique_angular
    shell = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    orbital = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert (ihelp.spread_ushell_to_shell(ushell) == shell).all()
    assert (ihelp.spread_ushell_to_orbital(ushell) == orbital).all()


def test_cart() -> None:
    unique = torch.tensor([1, 14])  # include sorting
    atom = torch.tensor([14, 1, 1])
    shell = torch.tensor([14, 14, 14, 1, 1, 1, 1])
    orbital = torch.tensor([14] * 9 + [1, 1, 1, 1])
    orbital_cart = torch.tensor([14] * 10 + [1, 1, 1, 1])

    angular = {
        1: [0, 0],  # H
        14: [0, 1, 2],  # Si
    }
    ihelp = IndexHelper.from_numbers(atom, angular)

    assert (ihelp.spread_uspecies_to_atom(unique) == atom).all()
    assert (ihelp.spread_uspecies_to_shell(unique) == shell).all()
    assert (ihelp.spread_uspecies_to_orbital(unique) == orbital).all()
    assert (ihelp.spread_uspecies_to_orbital_cart(unique) == orbital_cart).all()
    assert (ihelp.spread_atom_to_shell(atom) == shell).all()
    assert (ihelp.spread_atom_to_orbital(atom) == orbital).all()
    assert (ihelp.spread_atom_to_orbital_cart(atom) == orbital_cart).all()

    # check property
    orb_per_at = torch.tensor([0] * 9 + [1, 1, 2, 2])
    assert (ihelp.orbitals_per_atom == orb_per_at).all()

    # cartesian and spherical basis
    orb_per_shell = torch.tensor([1, 3, 5, 1, 1, 1, 1])
    orb_per_shell_cart = torch.tensor([1, 3, 6, 1, 1, 1, 1])

    orb_index = torch.tensor([0, 1, 4, 9, 10, 11, 12])
    orb_index_cart = torch.tensor([0, 1, 4, 10, 11, 12, 13])

    orb_to_shell = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6])
    orb_to_shell_cart = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6])

    assert (ihelp.orbital_index == orb_index).all()
    assert (ihelp.orbital_index_cart == orb_index_cart).all()

    assert (ihelp.orbitals_to_shell == orb_to_shell).all()
    assert (ihelp.orbitals_to_shell_cart == orb_to_shell_cart).all()

    assert (ihelp.orbitals_per_shell == orb_per_shell).all()
    assert (ihelp.orbitals_per_shell_cart == orb_per_shell_cart).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single(dtype: torch.dtype):
    numbers = symbol2number("S H H H Mg N O S N N C H C H O N".split())
    angular = {
        1: [0],  # H (GFN2!)
        6: [0, 1],  # C
        7: [0, 1],  # N
        8: [0, 1],  # O
        12: [0, 1, 2],  # Mg
        16: [0, 1, 2],  # S
    }

    ihelp = IndexHelper.from_numbers(numbers, angular)

    assert torch.sum(ihelp.angular >= 0) == torch.tensor(30)

    qat = torch.tensor(
        [
            *[-2.62608233282119e-1, +3.73633121487967e-1, +1.51424532948944e-1],
            *[+8.11274419840145e-2, +4.55582555217907e-1, +1.89469664895825e-1],
            *[-3.59350817183894e-1, -1.38911850317377e-1, -1.83689392824396e-1],
            *[-1.88906495161279e-1, +5.33440028285669e-2, +1.94112134916556e-1],
            *[+2.02080948377078e-1, +1.74595453525400e-1, -4.46124496927388e-1],
            *[-2.95778570663624e-1],
        ],
        dtype=dtype,
    )
    qsh = torch.tensor(
        [
            *[+5.72134559421376e-2, -2.68193499948548e-1, -5.17064903935052e-2],
            *[+3.73632853173886e-1, +1.51424477665324e-1, +8.11205008366953e-2],
            *[+1.05453876337982e-0, -4.64589774617786e-1, -1.34371775944868e-1],
            *[+3.21958772020979e-1, -1.32435004307411e-1, +2.88638899705825e-1],
            *[-6.47972813769995e-1, +6.82118177705109e-2, -1.10631364729565e-1],
            *[-9.64955180671905e-2, +1.27185941911165e-1, -3.10873201558534e-1],
            *[+9.97036415531523e-2, -2.88615729133477e-1, -1.09656595674679e-1],
            *[+1.63000176490660e-1, +1.94112048312228e-1, -4.48012133376332e-2],
            *[+2.46906120671464e-1, +1.74594629792853e-1, +2.06932598673206e-1],
            *[-6.52990922441455e-1, -6.98603488893812e-3, -2.88854759086314e-1],
        ],
        dtype=dtype,
    )

    assert pytest.approx(qat, abs=1.0e-4) == ihelp.reduce_shell_to_atom(qsh)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype):
    numbers = batch.pack(
        (
            symbol2number("O Al Si H Li H Cl Al H H B H H B H H".split()),
            symbol2number("H H Si H Na S H H Al H C Si Cl B B H".split()),
        )
    )
    angular = {
        1: [0],  # H (GFN2!)
        3: [0, 1],  # Li
        5: [0, 1],  # B
        6: [0, 1],  # C
        8: [0, 1],  # O
        11: [0, 1],  # Na
        13: [0, 1, 2],  # Al
        14: [0, 1, 2],  # Si
        16: [0, 1, 2],  # S
        17: [0, 1, 2],  # Cl
    }

    ihelp = IndexHelper.from_numbers(numbers, angular)

    assert torch.sum(ihelp.angular >= 0) == torch.tensor(58)

    qat = batch.pack(
        (
            torch.tensor(
                [
                    *[-5.35371225694038e-1, +1.27905155882876e-1, +2.06910619292535e-1],
                    *[-1.93061647443670e-1, +5.46833043573218e-1, +2.98577669101319e-1],
                    *[-3.62405585534705e-1, +2.07231134137244e-1, +2.85826164709174e-1],
                    *[-1.76518940177473e-1, +9.44972704818130e-2, -1.17451405142691e-1],
                    *[-1.41198286268662e-1, +1.05227974737201e-2, -1.31666840327078e-1],
                    *[-1.20629924063582e-1],
                ],
                dtype=dtype,
            ),
            torch.tensor(
                [
                    *[-5.41402496268596e-2, -1.33777153976276e-1, +4.14313829600631e-1],
                    *[-1.16641170075389e-1, +4.56021377424607e-1, -5.20378766868989e-1],
                    *[-1.63965423099635e-1, -7.65345311273482e-2, +2.08304494730413e-1],
                    *[-1.71827679329874e-1, -3.30458156481514e-1, +5.58638267294323e-1],
                    *[-3.10094747569162e-1, +1.56794592474036e-1, +2.13459748796815e-1],
                    *[-1.29714432165776e-1],
                ],
                dtype=dtype,
            ),
        )
    )
    qsh = batch.pack(
        (
            torch.tensor(
                [
                    *[+2.92177048596496e-1, -8.27551283559270e-1, +3.81811514612779e-1],
                    *[+4.17784916263666e-1, -6.71683789364610e-1, -5.11576334445887e-2],
                    *[+2.68488368058409e-1, -1.04391705441501e-2, -1.93062932974169e-1],
                    *[+7.61952673849415e-1, -2.15114295745990e-1, +2.98579359296096e-1],
                    *[+2.43594536377056e-2, -3.79828052486015e-1, -6.93861559657517e-3],
                    *[+9.40499498626918e-1, +3.55733525506819e-1, -1.08899859046528e-0],
                    *[+2.85827068858603e-1, -1.76521898491791e-1, +1.16110895118352e-0],
                    *[-1.06660369382576e-0, -1.17453146019547e-1, -1.41199843368407e-1],
                    *[+1.11757931464608e-0, -1.10704872438469e-0, -1.31668946428877e-1],
                    *[-1.20631076436794e-1],
                ],
                dtype=dtype,
            ),
            torch.tensor(
                [
                    *[-5.41503330947710e-2, -1.33779040525986e-1, +6.99211871952128e-2],
                    *[+5.19410210372243e-1, -1.75021497979320e-1, -1.16646288193380e-1],
                    *[+5.88579806566877e-1, -1.32544517759254e-1, +7.82133136453700e-4],
                    *[-5.23133226533954e-1, +1.95314227063571e-3, -1.63971802434084e-1],
                    *[-7.65416499768423e-2, +6.26092659280207e-1, +3.00350609071998e-1],
                    *[-7.18137148569625e-1, -1.71830433005059e-1, -1.31761941444373e-1],
                    *[-1.98713319565700e-1, +8.86264348375974e-2, +7.46929250011706e-1],
                    *[-2.76884353276538e-1, +4.27025703238206e-2, -3.52590124894769e-1],
                    *[-2.13102162478342e-4, +1.13142747674328e-0, -9.74609683930292e-1],
                    *[+1.02802427493832e-0, -8.14556120567196e-1, -1.29715170834719e-1],
                ],
                dtype=dtype,
            ),
        )
    )

    assert torch.allclose(
        qat,
        ihelp.reduce_shell_to_atom(qsh),
        atol=1.0e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch2(dtype: torch.dtype):
    numbers = batch.pack(
        (
            symbol2number("O Al Si H Li H Cl Al H H B H H B H H".split()),
            symbol2number("Si H H H H".split()),
        )
    )
    angular = {
        1: [0],  # H (GFN2!)
        3: [0, 1],  # Li
        5: [0, 1],  # B
        6: [0, 1],  # C
        8: [0, 1],  # O
        11: [0, 1],  # Na
        13: [0, 1, 2],  # Al
        14: [0, 1, 2],  # Si
        16: [0, 1, 2],  # S
        17: [0, 1, 2],  # Cl
    }

    ihelp = IndexHelper.from_numbers(numbers, angular)

    assert torch.sum(ihelp.angular >= 0) == torch.tensor(35)

    qat = batch.pack(
        (
            torch.tensor(
                [
                    *[-5.35371225694038e-1, +1.27905155882876e-1, +2.06910619292535e-1],
                    *[-1.93061647443670e-1, +5.46833043573218e-1, +2.98577669101319e-1],
                    *[-3.62405585534705e-1, +2.07231134137244e-1, +2.85826164709174e-1],
                    *[-1.76518940177473e-1, +9.44972704818130e-2, -1.17451405142691e-1],
                    *[-1.41198286268662e-1, +1.05227974737201e-2, -1.31666840327078e-1],
                    *[-1.20629924063582e-1],
                ],
                dtype=dtype,
            ),
            torch.tensor(
                [
                    *[+4.40703021483644e-1, -1.10175755370910e-1, -1.10175755370912e-1],
                    *[-1.10175755370911e-1, -1.10175755370911e-1],
                ],
                dtype=dtype,
            ),
        )
    )
    qsh = batch.pack(
        (
            torch.tensor(
                [
                    *[+2.92177048596496e-1, -8.27551283559270e-1, +3.81811514612779e-1],
                    *[+4.17784916263666e-1, -6.71683789364610e-1, -5.11576334445887e-2],
                    *[+2.68488368058409e-1, -1.04391705441501e-2, -1.93062932974169e-1],
                    *[+7.61952673849415e-1, -2.15114295745990e-1, +2.98579359296096e-1],
                    *[+2.43594536377056e-2, -3.79828052486015e-1, -6.93861559657517e-3],
                    *[+9.40499498626918e-1, +3.55733525506819e-1, -1.08899859046528e-0],
                    *[+2.85827068858603e-1, -1.76521898491791e-1, +1.16110895118352e-0],
                    *[-1.06660369382576e-0, -1.17453146019547e-1, -1.41199843368407e-1],
                    *[+1.11757931464608e-0, -1.10704872438469e-0, -1.31668946428877e-1],
                    *[-1.20631076436794e-1],
                ],
                dtype=dtype,
            ),
            torch.tensor(
                [
                    *[+1.10033553895207e-1, +5.54549482852800e-1, -2.23880015264363e-1],
                    *[-1.10175755370910e-1, -1.10175755370912e-1, -1.10175755370911e-1],
                    *[-1.10175755370911e-1],
                ],
                dtype=dtype,
            ),
        )
    )

    assert torch.allclose(
        qat,
        ihelp.reduce_shell_to_atom(qsh),
        atol=1.0e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch_2dim(dtype: torch.dtype) -> None:
    numbers = batch.pack(
        (
            symbol2number("H H".split()),
            symbol2number("Li H".split()),
        )
    )
    angular = {
        1: [0, 0],  # H
        3: [0, 1],  # Li
    }

    ihelp = IndexHelper.from_numbers(numbers, angular)

    ref = torch.tensor(
        [
            [
                [2.0000000000, 0.9030375481],
                [0.9030375481, 2.0000000000],
            ],
            [
                [4.0000000000, 0.5933173895],
                [0.5933173895, 2.0000000000],
            ],
        ],
        dtype=dtype,
    )

    s_h2 = torch.tensor(
        [
            [
                1.0000000000e00,
                8.5040041675e-10,
                6.6998297092e-01,
                6.5205745748e-02,
            ],
            [
                8.5040041675e-10,
                1.0000000000e00,
                6.5205745748e-02,
                1.0264305273e-01,
            ],
            [
                6.6998297092e-01,
                6.5205745748e-02,
                1.0000000000e00,
                8.5040041675e-10,
            ],
            [
                6.5205745748e-02,
                1.0264305273e-01,
                8.5040041675e-10,
                1.0000000000e00,
            ],
        ],
        dtype=dtype,
    )

    s_lih = torch.tensor(
        [
            [
                1.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                4.0560702913e-01,
                -2.0099517671e-01,
            ],
            [
                0.0000000000e00,
                1.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
            ],
            [
                0.0000000000e00,
                0.0000000000e00,
                1.0000000000e00,
                0.0000000000e00,
                4.6387245497e-01,
                -7.5166942324e-02,
            ],
            [
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                1.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
            ],
            [
                4.0560702913e-01,
                0.0000000000e00,
                4.6387245497e-01,
                0.0000000000e00,
                1.0000000000e00,
                8.5040041675e-10,
            ],
            [
                -2.0099517671e-01,
                0.0000000000e00,
                -7.5166942324e-02,
                0.0000000000e00,
                8.5040041675e-10,
                1.0000000000e00,
            ],
        ],
        dtype=dtype,
    )

    s = batch.pack([s_h2, s_lih])

    s_atom = ihelp.reduce_orbital_to_atom(s, dim=(-2, -1))
    assert pytest.approx(ref) == s_atom
