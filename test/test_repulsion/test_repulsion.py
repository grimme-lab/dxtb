
from typing import Callable, Union
from unittest import TestCase
import torch

from tbmalt.structures.geometry import Geometry
from xtbml.repulsion.repulsion import Repulsion
from .test_repulsion_data import data

from xtbml.data.covrad import to_number

""" Testing the calculation of repulsion energy and gradients. """

class Test_Repulsion(TestCase):

    @classmethod
    def setUpClass(cls):
        print("Test_Repulsion")

    def setUp(self):
        # define constants
        self.cutoff = 25.0
        self.atol = 1e-07
        self.rtol = 1e-07
    
    def repulsion_gfn1(self, geometry: Geometry) -> Repulsion:
        """Factory for repulsion construction based on GFN1-xTB"""

        # parameter library
        alpha_gfn1 = torch.tensor([2.209700, 1.382907, 0.671797, 0.865377, 1.093544, 1.281954, 1.727773, 2.004253, 2.507078, 3.038727, 0.704472, 0.862629, 0.929219, 0.948165, 1.067197, 1.200803, 1.404155, 1.323756, 0.581529, 0.665588])
        zeff_gfn1 = torch.tensor([1.116244, 0.440231, 2.747587, 4.076830, 4.458376, 4.428763, 5.498808, 5.171786, 6.931741, 9.102523, 10.591259, 15.238107, 16.283595, 16.898359, 15.249559, 15.100323, 17.000000, 17.153132, 20.831436, 19.840212])

        alpha = alpha_gfn1[geometry.unique_atomic_numbers()-1]
        zeff = zeff_gfn1[geometry.unique_atomic_numbers()-1]
        kexp = 1.5
        kexp_light = 1.5
        rexp = 1.0

        # setup repulsion
        repulsion = Repulsion(geometry=geometry)
        repulsion.setup(alpha, zeff, kexp, kexp_light, rexp)

        return repulsion
    
    def repulsion_gfn2(self, geometry: Geometry) -> Repulsion:
        """Factory for repulsion construction based on GFN2-xTB"""

        # parameter library
        alpha_gfn2 = torch.tensor([2.213717, 3.604670, 0.475307, 0.939696, 1.373856, 1.247655, 1.682689, 2.165712, 2.421394, 3.318479, 0.572728, 0.917975, 0.876623, 1.187323, 1.143343, 1.214553, 1.577144, 0.896198, 0.482206, 0.683051])
        zeff_gfn2 = torch.tensor([1.105388,1.094283,1.289367,4.221216,7.192431, 4.231078,5.242592,5.784415,7.021486, 11.041068, 5.244917, 18.083164, 17.867328, 40.001111, 19.683502,14.995090, 17.353134,7.266606, 10.439482, 14.786701])

        alpha = alpha_gfn2[geometry.unique_atomic_numbers()-1]
        zeff = zeff_gfn2[geometry.unique_atomic_numbers()-1]
        kexp = 1.5
        kexp_light = 1.0
        rexp = 1.0

        # setup repulsion
        repulsion = Repulsion(geometry=geometry)
        repulsion.setup(alpha, zeff, kexp, kexp_light, rexp)

        return repulsion
    
    def base_test(self, geometry: Geometry, repulsion_factory: Callable, reference_energy: Union[torch.Tensor, None], reference_gradient: Union[torch.Tensor, None]):
        """Wrapper for testing versus reference energy and gradient"""

        # factory to produce repulsion objects
        repulsion = repulsion_factory(geometry)
        energy, gradient = repulsion.get_engrad(geometry=geometry, cutoff=self.cutoff, calc_gradient=True)

        # test against reference values
        if reference_energy:
            self.assertTrue(
                torch.allclose(
                    energy, reference_energy, rtol=self.rtol, atol=self.atol, equal_nan=False
                ),
                msg=f"Energy not correct:\n {energy} vs. {reference_energy}",
            )
        if reference_gradient != None:
            self.assertTrue(
                torch.allclose(
                    gradient, reference_gradient, rtol=self.rtol, atol=self.atol, equal_nan=False
                ),
                msg=f"Gradient not correct:\n {gradient} vs. {reference_gradient}",
            )
        
        return
    
    def calc_numerical_gradient(self, geometry: Geometry, repulsion: Repulsion, cutoff: float):
        """Calculate gradient numerically for reference."""

        n_atoms = geometry.get_length(unique=False)

        # numerical gradient
        gradient = torch.zeros((n_atoms, 3))
        step = 1.0e-6

        for i in range(n_atoms):
            for j in range(3):
                er, el = 0.0, 0.0
                geometry.positions[i,j] += step
                er = repulsion.get_engrad(geometry, cutoff=cutoff, calc_gradient=False)
                geometry.positions[i,j] -= 2*step
                el = repulsion.get_engrad(geometry, cutoff=cutoff, calc_gradient=False)
                geometry.positions[i,j] += step
                gradient[i,j] = 0.5*(er - el)/step

        return gradient

    
    def test_mb1643_01_gfn1(self):
        sample = data["MB16_43_01"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = torch.tensor(0.16777923624986593)
        reference_gradient = None

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn1,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)
        
    def test_mb1643_02_gfn2(self):
        sample = data["MB16_43_02"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = torch.tensor(0.10745931926703985)
        reference_gradient = None

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)
        
    def test_mb1643_03_gfn1(self):
        sample = data["MB16_43_03"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        # get reference gradient
        repulsion = self.repulsion_gfn1(geometry)
        numerical_gradient = self.calc_numerical_gradient(geometry, repulsion, self.cutoff)

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn1,
                    reference_energy=None, 
                    reference_gradient=numerical_gradient)

    def test_mb1643_04_gfn2(self):
        sample = data["MB16_43_04"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        # get reference gradient
        repulsion = self.repulsion_gfn2(geometry)
        numerical_gradient = self.calc_numerical_gradient(geometry, repulsion, self.cutoff)

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=None, 
                    reference_gradient=numerical_gradient)



    # REQUIRES PBC implementation
    '''def test_uracil_gfn2(self):
        sample = data["uracil"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = torch.tensor(1.0401472262740301)
        reference_gradient = None
        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)'''
    
    # REQUIRES PBC implementation
    '''subroutine test_g_effective_urea(error)

    !> Error handling
    type(error_type), allocatable, intent(out) :: error

    type(structure_type) :: mol

    call get_structure(mol, "X23", "urea")
    call test_numgrad(error, mol, make_repulsion2)

    end subroutine test_g_effective_urea'''

# REQUIRES PBC implementation (strain and sigma not implemented yet)
'''
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

end module test_repulsion'''
