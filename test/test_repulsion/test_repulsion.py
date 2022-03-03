
from typing import Callable, Union
from unittest import TestCase
import torch
from math import sqrt
import sys

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
        # setup for each test

        # define constants
        self.cutoff = 25.0
        eps = sys.float_info.epsilon
        self.atol = 1e-05 # 100*eps
        self.rtol = 1e-01               # TODO: requires high tolerance
        #  TODO: check pytorch tensor accuraccy
        #        check why factor 1/2 is required in energy
        self.atol2 = 1e-05 # sqrt(eps)
        self.rtol2 = 1e-05
    
    def repulsion_gfn1(self, geometry: Geometry) -> Repulsion:
        """Factory for repulsion construction based on GFN1-xTB"""

        # parameter library
        alpha_gfn1 = torch.tensor([2.209700, 1.382907, 0.671797, 0.865377, 1.093544, 1.281954, 1.727773, 2.004253, 2.507078, 3.038727, 0.704472, 0.862629, 0.929219, 0.948165, 1.067197, 1.200803, 1.404155, 1.323756, 0.581529, 0.665588])
        zeff_gfn1 = torch.tensor([1.116244, 0.440231, 2.747587, 4.076830, 4.458376, 4.428763, 5.498808, 5.171786, 6.931741, 9.102523, 10.591259, 15.238107, 16.283595, 16.898359, 15.249559, 15.100323, 17.000000, 17.153132, 20.831436, 19.840212])

        alpha = alpha_gfn1[:geometry.get_length(unique=False)]
        zeff = zeff_gfn1[:geometry.get_length(unique=False)]
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

        alpha = alpha_gfn2[:geometry.get_length(unique=False)]
        zeff = zeff_gfn2[:geometry.get_length(unique=False)]
        kexp = 1.5
        kexp_light = 1.5
        rexp = 1.0

        # setup repulsion
        repulsion = Repulsion(geometry=geometry)
        repulsion.setup(alpha, zeff, kexp, kexp_light, rexp)

        return repulsion
    
    def base_test(self, geometry: Geometry, repulsion_factory: Callable, reference_energy: Union[torch.Tensor, None], reference_gradient: Union[torch.Tensor, None]):
        """Wrapper for testing"""

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
        if reference_gradient:
            self.assertTrue(
                torch.allclose(
                    gradient, reference_gradient, rtol=self.rtol, atol=self.atol, equal_nan=False
                ),
                msg=f"Gradient not correct:\n {gradient} vs. {reference_gradient}",
            )
        return

    
    def test_mb16_43_gfn1(self):
        """
        Compare against reference calculated with tblite-int H C 0,0,1.4 --bohr --method gfn1
        """

        sample = data["MB16_43_01"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = torch.tensor(0.16777923624986593)
        reference_gradient = None

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn1,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)
        
    '''def test_mb16_43_gfn2(self):
        """
        Compare against reference calculated with tblite-int H C 0,0,1.4 --bohr --method gfn1
        """

        sample = data["MB16_43_02"]

        atomic_numbers = torch.flatten(torch.tensor([to_number(s) for s in sample["elements"]]))
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["xyz"])

        reference_energy = torch.tensor(0.10745931926703985)
        reference_gradient = None

        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)'''



