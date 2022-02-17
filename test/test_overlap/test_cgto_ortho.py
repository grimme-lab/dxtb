
from unittest import TestCase
import torch

from xtbml.basis.type import Cgto_Type
from xtbml.basis.slater import slater_to_gauss
from xtbml.integral.overlap import overlap_cgto
from xtbml.basis.ortho import orthogonalize

""" Testing the orthogonality of orbital overlap. """

class Test_Cgto_Ortho(TestCase):

    @classmethod
    def setUpClass(cls):
        print("Test_Cgto_Ortho")
        
    def test_ortho_1s_2s(self):
        """ Test orthogonality of 1s and 2s orbitals"""

        # orbitals
        ng, l = 5, 0

        # same site 
        r2 = 0.0
        vec = torch.tensor([0., 0., 0.])

        # create gaussians
        cgtoi, cgtoj = Cgto_Type(), Cgto_Type()
        slater_to_gauss(5, 1, l, 1.0, cgtoi, True)
        slater_to_gauss(2, 2, l, 1.0, cgtoj, True)

        orthogonalize(cgtoi, cgtoj)

        # normalised self-overlap
        overlap = overlap_cgto(cgtoj, cgtoj, r2, vec, 100.0)
        self.assertTrue(torch.allclose(overlap, torch.eye(1), rtol=1e-05, atol=1e-05, equal_nan=False), msg=f"Self overlap not identity:\n {overlap}")
       
        # orthogonal overlap
        overlap = overlap_cgto(cgtoi, cgtoj, r2, vec, 100.0)
        self.assertTrue(torch.allclose(overlap, torch.zeros(1), rtol=1e-05, atol=1e-05, equal_nan=False), msg=f"Self overlap not identity:\n {overlap}")

        return
