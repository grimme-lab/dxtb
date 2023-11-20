"""
Libcint Interface
=================

This module provides the wrapper for interfacing with libcint.
The main class handles the conversion of from dxtb's basis format to libcint's
internal format.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

try:
    from dxtblibs import CINT  # type: ignore
except ImportError:
    pass

from ....._types import Iterator, Tensor
from .....basis import AtomCGTOBasis, CGTOBasis, IndexHelper
from .utils import NDIM, int2ctypes, memoize_method, np2ctypes

__all__ = ["LibcintWrapper"]

# Terminology:
# * gauss: one gaussian element (multiple gaussian becomes one shell)
# * shell: one contracted basis (the same contracted gaussian for different atoms
#          counted as different shells)
# * ao: shell that has been splitted into its components,
#       e.g. p-shell is splitted into 3 components for cartesian (x, y, z)

PTR_RINV_ORIG = 4  # from libcint/src/cint_const.h


class LibcintWrapper:
    """
    Wrapper for interfacing with libcint.

    The basis information is transformed in the format that libcint expects.
    """

    def __init__(
        self,
        atombases: list[AtomCGTOBasis],
        ihelp: IndexHelper,
        spherical: bool = True,
    ) -> None:
        self._atombases = atombases
        self._spherical = spherical
        self._fracz = False
        self._natoms = len(atombases)
        self.ihelp = ihelp

        # get dtype and device for torch's tensors
        self.dtype = atombases[0].bases[0].alphas.dtype
        self.device = atombases[0].bases[0].alphas.device

        # construct _atm, _bas, and _env as well as the parameters
        ptr_env = 20  # initial padding from libcint
        atm_list: list[list[int]] = []
        env_list: list[float] = [0.0] * ptr_env
        bas_list: list[list[int]] = []
        allpos: list[Tensor] = []
        allalphas: list[Tensor] = []
        allcoeffs: list[Tensor] = []
        allangmoms: list[int] = []
        shell_to_atom: list[int] = []
        ngauss_at_shell: list[int] = []

        # constructing the triplet lists and also collecting the parameters
        ishell = 0
        for iatom, atombasis in enumerate(atombases):
            # construct the atom environment
            assert atombasis.pos.numel() == NDIM, "Please report this bug in Github"
            atomz = atombasis.atomz
            #                charge    ptr_coord, nucl model (unused for standard nucl model)
            atm_list.append([int(atomz), ptr_env, 1, ptr_env + NDIM, 0, 0])
            env_list.extend(atombasis.pos.detach().tolist())
            env_list.append(0.0)
            ptr_env += NDIM + 1

            # check if the atomz is fractional
            if isinstance(atomz, float) or (
                isinstance(atomz, Tensor) and atomz.is_floating_point()
            ):
                self._fracz = True

            # add the atom position into the parameter list
            # TODO: consider moving allpos into shell
            allpos.append(atombasis.pos.unsqueeze(0))

            shell_to_atom.extend([iatom] * len(atombasis.bases))

            # then construct the basis
            for shell in atombasis.bases:
                assert (
                    shell.alphas.shape == shell.coeffs.shape and shell.alphas.ndim == 1
                ), "Please report this bug in Github"

                # WE DONT NORMALIZE!!!
                # shell.wfnormalize_()

                ngauss = len(shell.alphas)
                #                iatom, angmom,       ngauss, ncontr, kappa, ptr_exp
                bas_list.append(
                    [
                        iatom,
                        shell.angmom,
                        ngauss,
                        1,
                        0,
                        ptr_env,
                        # ptr_coeffs,           unused
                        ptr_env + ngauss,
                        0,
                    ]
                )
                env_list.extend(shell.alphas.detach().tolist())
                env_list.extend(shell.coeffs.detach().tolist())
                ptr_env += 2 * ngauss

                # add the alphas and coeffs to the parameters list
                allalphas.append(shell.alphas)
                allcoeffs.append(shell.coeffs)
                allangmoms.extend([shell.angmom] * ngauss)
                ngauss_at_shell.append(ngauss)
                ishell += 1

        # compile the parameters of this object
        self._allpos_params = torch.cat(allpos, dim=0)  # (natom, NDIM)
        self._allalphas_params = torch.cat(allalphas, dim=0)  # (ntot_gauss)
        self._allcoeffs_params = torch.cat(allcoeffs, dim=0)  # (ntot_gauss)
        self._allangmoms = torch.tensor(
            allangmoms, dtype=torch.int32, device=self.device
        )  # (ntot_gauss)

        # convert the lists to numpy to make it contiguous (Python lists are not contiguous)
        self._atm = np.array(atm_list, dtype=np.int32, order="C")
        self._bas = np.array(bas_list, dtype=np.int32, order="C")
        self._env = np.array(env_list, dtype=np.float64, order="C")

        # print("\nself._atm\n", self._atm)
        # print("\nself._bas\n", self._bas)
        # print("\nself._env\n", self._env)

        self._shell_idxs = (0, ihelp.nsh)  # nshells
        self._ngauss_at_shell_list = ngauss_at_shell

        if spherical is True:
            shell_to_aoloc = ihelp.orbital_index.tolist()
            shell_to_aoloc.append(torch.cumsum(ihelp.orbitals_per_shell, -1)[-1])
            self._shell_to_aoloc = np.array(shell_to_aoloc, dtype=np.int32)

            self._ao_to_shell = ihelp.orbitals_to_shell
            self._ao_to_atom = ihelp.orbitals_to_atom
        else:
            shell_to_aoloc = ihelp.orbital_index_cart.tolist()
            shell_to_aoloc.append(torch.cumsum(ihelp.orbitals_per_shell_cart, -1)[-1])
            self._shell_to_aoloc = np.array(shell_to_aoloc, dtype=np.int32)

            self._ao_to_shell = ihelp.orbitals_to_shell_cart
            self._ao_to_atom = ihelp.orbitals_to_atom_cart

        # print("")
        # print("self._shell_to_aoloc", self._shell_to_aoloc)
        # print("self._ao_to_atom", self._ao_to_atom)
        # print("self._ao_to_shell", self._ao_to_shell)
        # print("self._shell_idxs", self._shell_idxs)

    @property
    def parent(self) -> LibcintWrapper:
        # parent is defined as the full LibcintWrapper where it takes the full
        # shells for the integration (without the need for subsetting)
        return self

    @property
    def natoms(self) -> int:
        # return the number of atoms in the environment
        return self._natoms

    @property
    def fracz(self) -> bool:
        # indicating whether we are working with fractional z
        return self._fracz

    @property
    def spherical(self) -> bool:
        # returns whether the basis is in spherical coordinate (otherwise, it
        # is in cartesian coordinate)
        return self._spherical

    @property
    def atombases(self) -> list[AtomCGTOBasis]:
        return self._atombases

    @property
    def atm_bas_env(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns the triplet lists, i.e. atm, bas, env
        # this shouldn't change in the sliced wrapper
        return self._atm, self._bas, self._env

    @property
    def full_angmoms(self) -> Tensor:
        return self._allangmoms

    @property
    def params(self) -> tuple[Tensor, Tensor, Tensor]:
        # returns all the parameters of this object
        # this shouldn't change in the sliced wrapper
        return self._allcoeffs_params, self._allalphas_params, self._allpos_params

    @property
    def shell_idxs(self) -> tuple[int, int]:
        # returns the lower and upper indices of the shells of this object
        # in the absolute index (upper is exclusive)
        return self._shell_idxs

    @property
    def full_shell_to_aoloc(self) -> np.ndarray:
        # returns the full array mapping from shell index to absolute ao location
        # the atomic orbital absolute index of i-th shell is given by
        # (self.full_shell_to_aoloc[i], self.full_shell_to_aoloc[i + 1])
        # if this object is a subset, then returns the complete mapping
        return self._shell_to_aoloc

    @property
    def full_ao_to_atom(self) -> Tensor:
        # returns the full array mapping from atomic orbital index to the
        # atom location
        return self._ao_to_atom

    @property
    def full_ao_to_shell(self) -> Tensor:
        # returns the full array mapping from atomic orbital index to the
        # shell location
        return self._ao_to_shell

    @property
    def ngauss_at_shell(self) -> list[int]:
        # returns the number of gaussian basis at the given shell
        return self._ngauss_at_shell_list

    @memoize_method
    def __len__(self) -> int:
        # total shells
        return self.shell_idxs[-1] - self.shell_idxs[0]

    @memoize_method
    def nao(self) -> int:
        # returns the number of atomic orbitals
        shell_idxs = self.shell_idxs
        return (
            self.full_shell_to_aoloc[shell_idxs[-1]]
            - self.full_shell_to_aoloc[shell_idxs[0]]
        )

    @memoize_method
    def ao_idxs(self) -> tuple[int, int]:
        # returns the lower and upper indices of the atomic orbitals of this object
        # in the full ao map (i.e. absolute indices)
        shell_idxs = self.shell_idxs
        return (
            self.full_shell_to_aoloc[shell_idxs[0]],
            self.full_shell_to_aoloc[shell_idxs[1]],
        )

    @memoize_method
    def ao_to_atom(self) -> Tensor:
        # get the relative mapping from atomic orbital relative index to the
        # absolute atom position
        # this is usually used in scatter in backward calculation
        return self.full_ao_to_atom[slice(*self.ao_idxs())]

    @memoize_method
    def ao_to_shell(self) -> Tensor:
        # get the relative mapping from atomic orbital relative index to the
        # absolute shell position
        # this is usually used in scatter in backward calculation
        return self.full_ao_to_shell[slice(*self.ao_idxs())]

    @memoize_method
    def get_uncontracted_wrapper(self) -> tuple[LibcintWrapper, Tensor]:
        # returns the uncontracted LibcintWrapper as well as the mapping from
        # uncontracted atomic orbital (relative index) to the relative index
        # of the atomic orbital
        new_atombases = []
        for atombasis in self.atombases:
            atomz = atombasis.atomz
            pos = atombasis.pos
            new_bases = []
            for shell in atombasis.bases:
                angmom = shell.angmom
                alphas = shell.alphas
                coeffs = shell.coeffs
                normalized = shell.normalized
                new_bases.extend(
                    [
                        CGTOBasis(
                            angmom, alpha[None], coeff[None], normalized=normalized
                        )
                        for (alpha, coeff) in zip(alphas, coeffs)
                    ]
                )
            new_atombases.append(AtomCGTOBasis(atomz=atomz, bases=new_bases, pos=pos))
        uncontr_wrapper = LibcintWrapper(
            new_atombases, self.ihelp, spherical=self.spherical
        )

        # get the mapping uncontracted ao to the contracted ao
        uao2ao: list[int] = []
        idx_ao = 0
        # iterate over shells
        for i in range(len(self)):
            nao = self._nao_at_shell(i)
            uao2ao += list(range(idx_ao, idx_ao + nao)) * self.ngauss_at_shell[i]
            idx_ao += nao
        uao2ao_res = torch.tensor(uao2ao, dtype=torch.long, device=self.device)
        return uncontr_wrapper, uao2ao_res

    ############### misc functions ###############
    @contextmanager
    def centre_on_r(self, r: Tensor) -> Iterator:
        # set the centre of coordinate to r (usually used in rinv integral)
        # r: (ndim,)
        env = self.atm_bas_env[-1]
        prev_centre = env[PTR_RINV_ORIG : PTR_RINV_ORIG + NDIM]
        try:
            env[PTR_RINV_ORIG : PTR_RINV_ORIG + NDIM] = r.detach().numpy()
            yield
        finally:
            env[PTR_RINV_ORIG : PTR_RINV_ORIG + NDIM] = prev_centre

    def _nao_at_shell(self, sh: int) -> int:
        # returns the number of atomic orbital at the given shell index
        if self.spherical:
            op = CINT().CINTcgto_spheric
        else:
            op = CINT().CINTcgto_cart
        bas = self.atm_bas_env[1]
        return op(int2ctypes(sh), np2ctypes(bas))
