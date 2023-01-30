"""
The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch
from math import sqrt, pi

from .._types import Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..integral import mmd
from ..param import Param, get_elem_angular
from ..utils import batch, symmetrize, t2int, IntegralTransformError
from .mmd import sqrtpi3, nlm_cart, e_function, EFunction
from . import transform
import sys
import os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Overlap(TensorLike):
    """
    Overlap from atomic orbitals.
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp

    def build(self, positions: Tensor) -> Tensor:
        """Overlap calculation of unique shells pairs.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        if self.numbers.ndim > 1:
            o = []
            for _batch in range(self.numbers.shape[0]):
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `get_overlap`
                nums = batch.deflate(self.numbers[_batch])
                ihelp = IndexHelper.from_numbers(
                    nums, get_elem_angular(self.par.element)
                )

                bas = Basis(
                    torch.unique(nums),
                    self.par,
                    ihelp.unique_angular,
                    dtype=self.dtype,
                    device=self.device,
                )

                o.append(self.get_overlap(bas, positions[_batch], ihelp))

            overlap = batch.pack(o)
        else:
            bas = Basis(
                self.unique,
                self.par,
                self.ihelp.unique_angular,
                dtype=self.dtype,
                device=self.device,
            )
            overlap = self.get_overlap(bas, positions, self.ihelp)

        # force symmetry to avoid problems through numerical errors
        return symmetrize(overlap)

    def get_overlap(self, bas: Basis, positions: Tensor, ihelp: IndexHelper) -> Tensor:
        """Overlap calculation for a single molecule.

        Parameters
        ----------
        numbers : Tensor
            Unique atomic numbers of whole batch.
        positions : Tensor
            Positions of single molecule.

        Returns
        -------
        Tensor
            Overlap matrix for single molecule.
        """

        umap, n_unique_pairs = bas.unique_shell_pairs(ihelp)
        alphas, coeffs = bas.create_cgtos()

        # spread stuff to orbitals for indexing
        alpha = batch.index(
            batch.index(batch.pack(alphas), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )
        coeff = batch.index(
            batch.index(batch.pack(coeffs), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )
        positions = batch.index(
            batch.index(positions, ihelp.shells_to_atom),
            ihelp.orbitals_to_shell,
        )
        ang = ihelp.spread_shell_to_orbital(ihelp.angular)

        # overlap calculation
        ovlp = torch.zeros(*umap.shape, dtype=self.dtype, device=self.device)
        for uval in range(n_unique_pairs):
            pairs = self.get_pairs(umap, uval)
            first_pair = pairs[0]

            angi, angj = ang[first_pair]
            norbi = 2 * t2int(angi) + 1
            norbj = 2 * t2int(angj) + 1

            # collect [0, 0] entry of each subblock
            upairs = self.get_subblock_start(umap, uval, norbi, norbj)

            # we only require one pair as all have the same basis function
            alpha_tuple = (
                batch.deflate(alpha[first_pair][0]),
                batch.deflate(alpha[first_pair][1]),
            )
            coeff_tuple = (
                batch.deflate(coeff[first_pair][0]),
                batch.deflate(coeff[first_pair][1]),
            )
            ang_tuple = (angi, angj)

            vec = positions[upairs][:, 0, :] - positions[upairs][:, 1, :]
            stmp = mmd.overlap(ang_tuple, alpha_tuple, coeff_tuple, -vec)

            # write overlap of unique pair to correct position in full overlap matrix
            for r, pair in enumerate(upairs):
                ovlp[
                    pair[0] : pair[0] + norbi,
                    pair[1] : pair[1] + norbj,
                ] = stmp[r]

        return ovlp  # .fill_diagonal_(1.0)

    def get_pairs(self, x: Tensor, i: int) -> Tensor:
        """Get indices of all unqiue shells pairs with index value `i`.

        Parameters
        ----------
        x : Tensor
            Matrix of unique shell pairs.
        i : int
            Value representing all unique shells in the matrix.

        Returns
        -------
        Tensor
            Indices of all unique shells pairs with index value `i` in the matrix.
        """

        return (x == i).nonzero(as_tuple=False)

    def get_subblock_start(
        self, umap: Tensor, i: int, norbi: int, norbj: int
    ) -> Tensor:
        """
        Filter out the top-left index of each subblock of unique shell pairs. This makes use of the fact that the pairs are sorted along
        the rows.

        Example: A "s" and "p" orbital would give the following 4x4 matrix
        of unique shell pairs:
        1 2 2 2
        3 4 4 4
        3 4 4 4
        3 4 4 4
        As the overlap routine gives back tensors of the shape `(norbi, norbj)`,
        i.e. 1x1, 1x3, 3x1 and 3x3 here, we require only the following four
        indices from the matrix of unique shell pairs: [0, 0] (1x1), [1, 0]
        (3x1), [0, 1] (1x3) and [1, 1] (3x3).


        Parameters
        ----------
        pairs : Tensor
            Indices of all unique shell pairs of one type (n, 2).
        norbi : int
            Number of orbitals per shell.
        norbj : int
            Number of orbitals per shell.

        Returns
        -------
        Tensor
            Top-left (i.e. [0, 0]) index of each subblock.
        """

        # no need to filter out a 1x1 block
        if norbi == 1 and norbj == 1:
            return self.get_pairs(umap, i)

        # sorting along rows allows only selecting every `norbj`th pair
        if norbi == 1:
            pairs = self.get_pairs(umap, i)
            return pairs[::norbj]

        if norbj == 1:
            pairs = self.get_pairs(umap.mT, i)

            # do the same for the transposed pairs, but switch columns
            return torch.index_select(pairs[::norbi], 1, torch.tensor([1, 0]))

        # more intricate because we can have variation in two dimensions
        pairs = self.get_pairs(umap, i)

        # remove every `norbj`th pair as before; only second dim is tricky
        pairs = pairs[::norbj]

        start = 0
        rest = pairs

        # init with dummy
        final = torch.tensor([[-1, -1]])

        while True:
            # get number of blocks in a row by counting the number of
            # same indices in the first dimension
            nb = (pairs[:, 0] == pairs[start, 0]).nonzero().flatten().size(0)

            # we need to skip the amount of rows in the block
            skip = nb * norbi

            # split for the blocks in each row because there can be different numbers of blocks in a row
            target, rest = torch.split(rest, [skip, rest.size(-2) - skip])

            # select only the top left index of each block
            final = torch.cat((final, target[:nb]), 0)

            start += skip
            if start >= pairs.size(-2):
                break

        # remove dummy
        return final[1:]

    def get_overlap_grad(
        self, bas: Basis, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """Overlap gradient dS/dr for a single molecule.

        Args:
            bas (Basis): Basis set for calculation.
            positions (Tensor): Positions of single molecule.
            ihelp (IndexHelper): Index helper for orbital mapping.

        Returns:
            Tensor: Gradient of overlap for single molecule.
        """

        umap, n_unique_pairs = bas.unique_shell_pairs(ihelp)
        alphas, coeffs = bas.create_cgtos()

        # spread stuff to orbitals for indexing
        alpha = batch.index(
            batch.index(batch.pack(alphas), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )
        coeff = batch.index(
            batch.index(batch.pack(coeffs), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )
        positions = batch.index(
            batch.index(positions, ihelp.shells_to_atom),
            ihelp.orbitals_to_shell,
        )
        ang = ihelp.spread_shell_to_orbital(ihelp.angular)

        # overlap tensor
        grad = torch.zeros((3, *umap.shape), dtype=self.dtype, device=self.device)

        # loop over unique pairs
        for uval in range(n_unique_pairs):
            pairs = self.get_pairs(umap, uval)
            first_pair = pairs[0]

            li, lj = ang[first_pair]
            norbi = 2 * t2int(li) + 1
            norbj = 2 * t2int(lj) + 1

            # collect [0, 0] entry of each subblock
            upairs = self.get_subblock_start(umap, uval, norbi, norbj)

            # we only require one pair as all have the same basis function
            alpha_tuple = (
                batch.deflate(alpha[first_pair][0]),
                batch.deflate(alpha[first_pair][1]),
            )
            coeff_tuple = (
                batch.deflate(coeff[first_pair][0]),
                batch.deflate(coeff[first_pair][1]),
            )
            ang_tuple = (li, lj)

            vecs = positions[upairs][:, 0, :] - positions[upairs][:, 1, :]

            # TODO: loop over atoms instead of unique (difficult to match contributions)
            # TODO: vectorize w.r.t. multiple vectors
            # stmp = mmd.overlap(ang_tuple, alpha_tuple, coeff_tuple, -vecs)
            tmp_ = []
            for vec in vecs:
                print("vec", vec, vec.shape)
                with HiddenPrints():
                    # TODO: +vec or -vec
                    tmp_.append(
                        get_ovlp_grad(ang_tuple, alpha_tuple, coeff_tuple, -vec)
                    )

            stmp = torch.stack(tmp_)
            print("stmp_final", stmp.shape)

            # write overlap of unique pair to correct position in full overlap matrix
            for r, pair in enumerate(upairs):
                # print("pair[0]", pair[0], pair[0] + norbi)
                # print("pair[1]", pair[1], pair[1] + norbj)
                # print("ovlp", ovlp[:, pair[0] : pair[0] + norbi,pair[1] : pair[1] + norbj].shape)
                # print("stmp[r]", stmp[r].shape)

                # TODO: check correct assignment
                grad[:, pair[0] : pair[0] + norbi, pair[1] : pair[1] + norbj,] = stmp[
                    r
                ]  # both [3, norbi, norbj]

        torch.set_printoptions(precision=2, linewidth=150, sci_mode=False)
        print("grad", grad)  # [3, norb, norb]

        return grad  # .fill_diagonal_(1.0)

    def get_gradient(self, positions: Tensor) -> Tensor:

        print("self.numbers.ndim > 1", self.numbers.ndim > 1)

        bas = Basis(
            self.unique,
            self.par,
            self.ihelp.unique_angular,
            dtype=self.dtype,
            device=self.device,
        )

        # correct implementation (TODO: sorting of entries might be incorrect)
        dsdr = self.get_overlap_grad(bas, positions, self.ihelp)  # [3, norb, norb]

        print("dsdr", dsdr)
        print("dsdr", dsdr.shape)

        # TODO: convert from [3, norb, norb] -> [3, natm] ?
        abc0 = self.ihelp.reduce_orbital_to_atom(dsdr[0], dim=(-2, -1))
        abc1 = self.ihelp.reduce_orbital_to_atom(dsdr[1], dim=(-2, -1))
        abc2 = self.ihelp.reduce_orbital_to_atom(dsdr[2], dim=(-2, -1))  # [natm, natm]
        print("abc0", abc0, abc0.shape)
        print("abc1", abc1, abc1.shape)
        print("abc2", abc2, abc2.shape)
        # maybe: [3, norb, norb] -> [3, natm, natm] -> [3, natm, natm] * [natm] -> [3, natm] (JVP)

        print("Today only calculating overlap (no gradient for you).")
        # sys.exit(0)

        # full jacobian shape: [norb, norb, natm, 3]
        # finally, the contracted gradient should be [natm, 3]
        return dsdr


def e_function_tblite_mmd_derivative(e, ai, lj, li):
    maxl = 6
    maxl2 = 2 * maxl  # TODO: infer correct shape

    de = torch.zeros((maxl2 + 1, maxl + 1, maxl + 1, 3))

    for m in range(3):  # TODO: check indices correct
        for i in range(li + 1):
            for j in range(lj + 1):
                de[0, j, i, m] = 2 * ai * e[0, j, i + 1, m]
                if i > 0:
                    de[0, j, i, m] -= i * e[0, j, i - 1, m]

    return de


def get_ovlp_grad(angular: tuple, alpha: tuple, coeff: tuple, vec: Tensor):
    """Implementation identical to tblite overlap gradient."""

    from dxtb.integral.mmd import sqrtpi3, nlm_cart, _e_function
    from dxtb.integral import transform
    from dxtb.utils import IntegralTransformError

    verbose = False

    def e_function(lj, li, xij, rpj, rpi):

        shape = torch.zeros((3, li + 1, lj + 1, li + lj + 2))
        print("inside e fct", shape.shape, xij.shape, rpi.shape, rpj.shape)
        E = _e_function(shape, xij, rpi, rpj)

        # TODO: adapt order of rest (more pythonic instead of fortran style)
        E = E.permute(3, 2, 1, 0)
        return E

    # number of primitive gaussians
    nprim_i = len(alpha[0].nonzero(as_tuple=True)[0])
    nprim_j = len(alpha[1].nonzero(as_tuple=True)[0])

    # angular momenta and number of cartesian gaussian basis functions
    li, lj = angular
    ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")

    s3d = torch.zeros([ncarti, ncartj])
    ds3d = torch.zeros([3, ncarti, ncartj])

    r2 = torch.sum(vec.pow(2), -1)
    if verbose:
        print("vec", vec)
        print("r2", r2)

    for ip in range(nprim_i):
        for jp in range(nprim_j):

            ai, aj = alpha[0][ip], alpha[1][jp]
            ci, cj = coeff[0][ip], coeff[1][jp]
            eab = ai + aj
            oab = 1.0 / eab
            est = ai * aj * r2 * oab
            cc = torch.exp(-est) * sqrtpi3 * torch.pow(oab, 1.5) * ci * cj  # sij

            rpi = -vec * aj * oab
            rpj = +vec * ai * oab

            et = e_function(lj + 1, li + 1, 0.5 * oab, rpj, rpi)
            # print("et", torch.nonzero(et), et.shape)

            det = e_function_tblite_mmd_derivative(et, ai, lj, li)
            # print("det", torch.nonzero(det), det.shape)

            for mli in range(ncarti):
                for mlj in range(ncartj):
                    mi = nlm_cart[li][mli, :]  # [3]
                    mj = nlm_cart[lj][mlj, :]  # [3]

                    if verbose:
                        print("li, lj", li, lj)
                        print("mi", mi, mi.shape)
                        print("mj", mj, mj.shape)

                    # TODO: check indices of et and det
                    if verbose:
                        print("et", et.shape)
                    e0 = torch.tensor(
                        [
                            et[0, mj[0], mi[0], 0],
                            et[0, mj[1], mi[1], 1],
                            et[0, mj[2], mi[2], 2],
                        ]
                    )
                    d0 = torch.tensor(
                        [
                            det[0, mj[0], mi[0], 0],
                            det[0, mj[1], mi[1], 1],
                            det[0, mj[2], mi[2], 2],
                        ]
                    )
                    if verbose:
                        print("e0", e0, e0.shape)
                        print("d0", d0, d0.shape)

                    s3d[mli, mlj] += cc * torch.prod(e0)

                    if verbose:
                        print("s3d", s3d[mli, mlj])
                        print(
                            [
                                d0[0] * e0[1] * e0[2],
                                e0[0] * d0[1] * e0[2],
                                e0[0] * e0[1] * d0[2],
                            ]
                        )

                    grad = torch.tensor(
                        [
                            d0[0] * e0[1] * e0[2],
                            e0[0] * d0[1] * e0[2],
                            e0[0] * e0[1] * d0[2],
                        ]
                    )
                    ds3d[:, mli, mlj] += cc * grad
                    if verbose:
                        print("grad", grad)
                        print("cc", cc)

    torch.set_printoptions(precision=4, sci_mode=False)
    print("At end")
    print("s3d", s3d)
    print("ds3d", ds3d)  # [3, 6, 6]
    print("s3d", s3d.shape)
    print("ds3d", ds3d.shape)

    # NOTE: sorting of s3d, ds3d is different in tblite <-> dxtb (fixed by transformations)

    # transform from cartesian to spherical gaussians
    try:
        itrafo = transform.trafo[li].type(s3d.dtype).to(s3d.device)
        jtrafo = transform.trafo[lj].type(s3d.dtype).to(s3d.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    print("here we are", li, lj)
    print("itrafo", itrafo)
    print("jtrafo", jtrafo)
    print("s3d.shape", s3d.shape)
    # s3d = torch.transpose(s3d, -1, -2) # TODO need to transpose due to ordering of mj, ml
    print("s3d.shape", s3d.shape)

    import sys

    # sys.exit(0)

    # test/test_overlap/test_grad.py::test_overlap_grad_single_5[dtype0] ovlp
    # tensor([[-0.0069032917, -0.1491720825,  0.0012659596],
    #         [-0.0284731276,  0.0024775318, -0.0557287671],
    #         [ 0.0943140090, -0.0135100093, -0.0284731276],
    #         [-0.0078950552, -0.1801633537,  0.0039291698],
    #         [-0.0043234574, -0.0683780164, -0.0057450016],
    #         [-0.1020899490,  0.0033691206, -0.1868773401],
    #         [ 0.0863762349, -0.0055685295, -0.1402334571]]) torch.Size([7, 3])
    # ovlp_ref
    # tensor([[ 0.0863759965, -0.0055689998, -0.1402339935],
    #         [-0.0043230001, -0.0683780015, -0.0057450002],
    #         [ 0.0943140015, -0.0135100000, -0.0284729991],
    #         [-0.0069030002, -0.1491719931,  0.0012660000],
    #         [-0.0284729991,  0.0024780000, -0.0557290018],
    #         [-0.0078950003, -0.1801629961,  0.0039289999],
    #         [-0.1020900011,  0.0033690000, -0.1868769974]]) torch.Size([7, 3])

    # 3           7
    # ovlp_mmd
    # 0.086376 -0.005569 -0.140234 -0.004323 -0.068378 -0.005745  0.094314
    # -0.013510 -0.028473 -0.006903 -0.149172  0.001266 -0.028473  0.002478
    # -0.055729 -0.007895 -0.180163  0.003929 -0.102090  0.003369 -0.186877

    # 7           3
    # ovlp_mmd
    # 0.086376 -0.004323  0.094314 -0.006903 -0.028473 -0.007895 -0.102090
    # -0.005569 -0.068378 -0.013510 -0.149172  0.002478 -0.180163  0.003369
    # -0.140234 -0.005745 -0.028473  0.001266 -0.055729  0.003929 -0.186877

    # transform overlap to cartesian basis functions (itrafo^T * S * jtrafo)
    ovlp = torch.einsum("...ji,...jk,...kl->...il", itrafo, s3d, jtrafo)
    print("final overlap\n", ovlp)
    print("final overlap\n", ovlp.shape)

    print("reshaping")
    print("ovlp.shape", ovlp.shape)
    ovlp1 = ovlp.permute(1, 0)
    print("ovlp1.shape", ovlp1, ovlp1.shape)
    # ovlp2 = ovlp.permute(2,1)
    # print("ovlp2.shape", ovlp2, ovlp2.shape)

    # transform gradient to spherical basis functions
    cart = ds3d
    sphr = [
        torch.einsum("...ji,...jk,...kl->...il", itrafo, cart[k, :, :], jtrafo)
        for k in range(cart.shape[-3])
    ]
    sphr = torch.stack(sphr)  # [3, norb, norb]

    return sphr


# TODO:
# 1. check for different cgtos and vectors that tblite and dxtb s + ds are identical
# 2. cleanup the function here (keep reference values)
# 3. write some test to check identical behaviour
# 3. implement as nice backward function into main workflow
