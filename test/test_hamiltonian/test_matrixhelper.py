import torch

from xtbml.basis import IndexHelper, MatrixHelper, get_elem_param_shells
from xtbml.param.gfn1 import GFN1_XTB as par

# TEST matricies (range matrix for H2O example)
test_matrix = torch.arange(64).reshape(8, 8)

water_hamiltonian = torch.tensor(
    [
        [-0.8701, 0.0000, 0.0000, 0.0000, -0.4655, -0.0501, -0.4655, -0.0501],
        [0.0000, -0.6534, 0.0000, 0.0000, 0.3348, 0.1569, -0.3348, -0.1569],
        [0.0000, 0.0000, -0.6534, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, -0.6534, -0.2601, -0.1218, -0.2601, -0.1218],
        [-0.4655, 0.3348, 0.0000, -0.2601, -0.4038, 0.0000, -0.1897, 0.0645],
        [-0.0501, 0.1569, 0.0000, -0.1218, 0.0000, -0.0803, 0.0645, -0.0126],
        [-0.4655, -0.3348, 0.0000, -0.2601, -0.1897, 0.0645, -0.4038, 0.0000],
        [-0.0501, -0.1569, 0.0000, -0.1218, 0.0645, -0.0126, 0.0000, -0.0803],
    ]
)
water_number = torch.tensor([8, 1, 1])

# anuglar momenta and respective orbitals for different elements
angular, _ = get_elem_param_shells(par.element, valence=True)

ihelp = IndexHelper.from_numbers(water_number, angular)


def test_matrixhelper_orbital_columns():

    columns = MatrixHelper.get_orbital_columns(test_matrix, ihelp)

    test_columns = (
        torch.tensor(
            [
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [16, 17, 18, 19],
                [24, 25, 26, 27],
                [32, 33, 34, 35],
                [40, 41, 42, 43],
                [48, 49, 50, 51],
                [56, 57, 58, 59],
            ]
        ),
        torch.tensor(
            [
                [4, 5],
                [12, 13],
                [20, 21],
                [28, 29],
                [36, 37],
                [44, 45],
                [52, 53],
                [60, 61],
            ]
        ),
        torch.tensor(
            [
                [6, 7],
                [14, 15],
                [22, 23],
                [30, 31],
                [38, 39],
                [46, 47],
                [54, 55],
                [62, 63],
            ]
        ),
    )

    assert len(columns) == list(water_number.shape)[0]
    assert all(
        [j[0] == len(ihelp.orbitals_to_shell) for j in [list(i.shape) for i in columns]]
    )

    coleq = [torch.equal(column, test_columns[i]) for i, column in enumerate(columns)]
    assert all(coleq)


def test_matrixhelper_orbital_sum():

    sums = MatrixHelper.get_orbital_sum(test_matrix, ihelp)

    test_sums = (
        torch.tensor([28, 92, 156, 220]),
        torch.tensor([284, 348]),
        torch.tensor([412, 476]),
    )

    sumeq = [torch.equal(s, test_sums[i]) for i, s in enumerate(sums)]
    assert all(sumeq)


def test_matrixhelper_atomblock():

    # NOTE: only possible for given orbital structure
    block = MatrixHelper.get_atomblock(test_matrix, 0, 1, ihelp)
    assert torch.equal(block, test_matrix[0:4, 4:6])

    block = MatrixHelper.get_atomblock(test_matrix, 0, 2, ihelp)
    assert torch.equal(block, test_matrix[0:4, 6:8])

    block = MatrixHelper.get_atomblock(test_matrix, 1, 2, ihelp)
    assert torch.equal(block, test_matrix[4:6, 6:8])


def test_matrixhelper_diagonal_blocks():

    diag_blocks = MatrixHelper.get_diagonal_blocks(test_matrix, ihelp)

    test_diag_blocks = (
        test_matrix[0:4, 0:4],
        test_matrix[4:6, 4:6],
        test_matrix[6:8, 6:8],
    )

    diageq = [torch.equal(db, test_diag_blocks[i]) for i, db in enumerate(diag_blocks)]
    assert all(diageq)
