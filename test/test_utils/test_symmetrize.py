"""
Test the utility functions.
"""
import pytest
import torch

from dxtb.utils import symmetrize


def test_fail() -> None:
    with pytest.raises(RuntimeError):
        symmetrize(torch.tensor([1, 2, 3]))

    with pytest.raises(RuntimeError):
        mat = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 2.0],
                [4.0, 3.0, 3.0],
            ]
        )
        symmetrize(mat)


def test_success() -> None:
    mat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 3.0],
        ]
    )

    sym = symmetrize(mat)
    assert (mat == sym).all()
    assert (sym == sym.mT).all()

    # make unsymmetric
    mat[0, -1] += 1e-5
    assert not (mat == mat.mT).all()

    # now symmetrize
    sym = symmetrize(mat)
    assert (sym == sym.mT).all()
