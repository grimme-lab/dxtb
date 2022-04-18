from typing import Callable, List, Union
from unittest import TestCase
import torch
from torch import Tensor

from xtbml.exlibs.tbmalt import Geometry, batch
from xtbml.repulsion.repulsion import Repulsion
from xtbml.utils import symbol2number

from .test_repulsion_data import data

from xtbml.param.gfn1 import GFN1_XTB

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TestRepulsion(TestCase):
    """Testing the calculation of repulsion energy and gradients."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def setUp(self):
        # define constants
        self.cutoff = 25.0
        self.atol = 1e-07
        self.rtol = 1e-07

    def repulsion_gfn1(self, geometry: Geometry) -> Repulsion:
        """Factory for repulsion construction based on GFN1-xTB"""

        # setup repulsion
        repulsion = Repulsion(geometry=geometry)
        repulsion.setup(GFN1_XTB.element, GFN1_XTB.repulsion.effective)

        return repulsion

    def repulsion_gfn2(self, geometry: Geometry) -> Repulsion:
        """Factory for repulsion construction based on GFN2-xTB"""

        # setup repulsion
        repulsion = Repulsion(geometry=geometry)
        # repulsion.setup(GFN2_XTB.element, GFN2_XTB.repulsion.effective)

        return repulsion

    def base_test(
        self,
        geometry: Geometry,
        repulsion_factory: Callable,
        reference_energy: Union[List[Tensor], Tensor, None],
        reference_gradient: Union[Tensor, None],
    ):
        """Wrapper for testing versus reference energy and gradient"""

        # factory to produce repulsion objects
        repulsion = repulsion_factory(geometry)
        energy, gradient = repulsion.get_engrad(
            geometry=geometry, cutoff=self.cutoff, calc_gradient=True
        )

        # test against reference values
        if reference_energy is not None:
            self.assertTrue(
                torch.allclose(
                    energy,
                    reference_energy,
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=False,
                ),
                msg=f"Energy not correct:\n {energy} vs. {reference_energy}",
            )
        if reference_gradient is not None:
            self.assertTrue(
                torch.allclose(
                    gradient,
                    reference_gradient,
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=False,
                ),
                msg=f"Gradient not correct:\n {gradient} vs. {reference_gradient}",
            )

        return

    def calc_numerical_gradient(
        self, geometry: Geometry, repulsion: Repulsion, cutoff: float
    ):
        """Calculate gradient numerically for reference."""

        n_atoms = geometry.get_length(unique=False)

        # numerical gradient
        gradient = torch.zeros((n_atoms, 3))
        step = 1.0e-6

        for i in range(n_atoms):
            for j in range(3):
                er, el = 0.0, 0.0
                geometry.positions[i, j] += step
                er = repulsion.get_engrad(geometry, cutoff=cutoff, calc_gradient=False)
                geometry.positions[i, j] -= 2 * step
                el = repulsion.get_engrad(geometry, cutoff=cutoff, calc_gradient=False)
                geometry.positions[i, j] += step
                gradient[i, j] = 0.5 * (er - el) / step

        return gradient

    def test_mb1643_01_gfn1(self):
        sample = data["MB16_43_01"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["xyz"])

        reference_energy = torch.tensor(0.16777923624986593)
        reference_gradient = None

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
        )

    def test_mb1643_02_gfn1(self):
        sample = data["MB16_43_02"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["xyz"])

        reference_energy = torch.tensor(0.12702003611285190)
        reference_gradient = None

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
        )

    def test_mb1643_03_gfn1(self):
        sample = data["MB16_43_03"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["xyz"])

        # get reference gradient
        repulsion = self.repulsion_gfn1(geometry)
        numerical_gradient = self.calc_numerical_gradient(
            geometry, repulsion, self.cutoff
        )

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=None,
            reference_gradient=numerical_gradient,
        )

    def stest_gfn1_batch(self):
        sample1, sample2 = data["MB16_43_01"], data["SiH4"]
        geometry = Geometry(
            batch.pack(
                (
                    symbol2number(sample1["elements"]),
                    symbol2number(sample2["elements"]),
                )
            ),
            batch.pack(
                (
                    sample1["xyz"],
                    sample2["xyz"],
                )
            ),
        )

        reference_energy = torch.tensor([0.16777923624986593, 0.12702003611285190])
        reference_gradient = None

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
        )

    def stest_mb1643_02_gfn2(self):
        sample = data["MB16_43_02"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["xyz"])

        reference_energy = torch.tensor(0.10745931926703985)
        reference_gradient = None

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn2,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
        )

    def stest_mb1643_04_gfn2(self):
        sample = data["MB16_43_04"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["xyz"])

        # get reference gradient
        repulsion = self.repulsion_gfn2(geometry)
        numerical_gradient = self.calc_numerical_gradient(
            geometry, repulsion, self.cutoff
        )

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn2,
            reference_energy=None,
            reference_gradient=numerical_gradient,
        )

    # REQUIRES PBC implementation
    """def test_uracil_gfn2(self):
        sample = data["uracil"]

        atomic_numbers = symbol2number(sample["elements"])
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = Tensor(1.0401472262740301)
        reference_gradient = None
        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)"""

    # REQUIRES PBC implementation
    """subroutine test_g_effective_urea(error)

    !> Error handling
    type(error_type), allocatable, intent(out) :: error

    type(structure_type) :: mol

    call get_structure(mol, "X23", "urea")
    call test_numgrad(error, mol, make_repulsion2)

    end subroutine test_g_effective_urea"""


# REQUIRES PBC implementation (strain and sigma not implemented yet)
"""
subroutine test_s_effective_m05(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "MB16-43", "05")
   call test_numsigma(error, mol, make_repulsion1)

end subroutine test_s_effective_m05


subroutine test_s_effective_m06(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "MB16-43", "06")
   call test_numsigma(error, mol, make_repulsion2)

end subroutine test_s_effective_m06


subroutine test_s_effective_succinic(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "X23", "succinic")
   call test_numsigma(error, mol, make_repulsion2)

end subroutine test_s_effective_succinic

end module test_repulsion"""
