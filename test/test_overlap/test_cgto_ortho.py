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
        """Test orthogonality of 1s and 2s orbitals"""

        # orbitals
        ng, l = 5, 0

        # same site
        r2 = 0.0
        vec = torch.tensor([0.0, 0.0, 0.0])

        # create gaussians
        cgtoi, cgtoj = Cgto_Type(), Cgto_Type()
        slater_to_gauss(5, 1, l, 1.0, cgtoi, True)
        slater_to_gauss(2, 2, l, 1.0, cgtoj, True)

        orthogonalize(cgtoi, cgtoj)

        # normalised self-overlap
        overlap = overlap_cgto(cgtoj, cgtoj, r2, vec, 100.0)
        self.assertTrue(
            torch.allclose(
                overlap, torch.eye(1), rtol=1e-05, atol=1e-05, equal_nan=False
            ),
            msg=f"Self overlap not identity:\n {overlap}",
        )

        # orthogonal overlap
        overlap = overlap_cgto(cgtoi, cgtoj, r2, vec, 100.0)
        self.assertTrue(
            torch.allclose(
                overlap, torch.zeros(1), rtol=1e-05, atol=1e-05, equal_nan=False
            ),
            msg=f"Self overlap not identity:\n {overlap}",
        )

        return

    def test_overlap_h_c(self):
        """
        Compare against reference calculated with tblite-int H C 0,0,1.4 --bohr --method gfn1
        """
        from xtbml.param.gfn1 import GFN1_XTB as par
        from xtbml.xtb.calculator import Basis

        basis = Basis(["H", "C"], par)
        h = basis.cgto.get("H")
        c = basis.cgto.get("C")

        vec = torch.tensor([0.0, 0.0, 1.4])
        r2 = vec.dot(vec)

        ref = [
            torch.tensor([[+6.77212228e-01]]),
            torch.tensor([[+0.00000000e-00], [+0.00000000e-00], [-5.15340812e-01]]),
            torch.tensor([[+7.98499991e-02]]),
            torch.tensor([[+0.00000000e-00], [+0.00000000e-00], [-1.72674504e-01]]),
        ]

        for ish in h:
            for jsh in c:
                overlap = overlap_cgto(ish, jsh, r2, vec, 100.0)
                self.assertTrue(
                    torch.allclose(
                        overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
                    ),
                    msg=f"Overlap does not match:\n {overlap}",
                )

    def test_overlap_h_he(self):
        """
        Compare against reference calculated with tblite-int H He 0,0,1.7 --method gfn1 --bohr
        """
        from xtbml.param.gfn1 import GFN1_XTB as par
        from xtbml.xtb.calculator import Basis

        basis = Basis(["H", "He"], par)
        h = basis.cgto.get("H")
        he = basis.cgto.get("He")

        vec = torch.tensor([0.0, 0.0, 1.7])
        r2 = vec.dot(vec)

        ref = [
            torch.tensor([[3.67988976e-01]]),
            torch.tensor([[9.03638642e-02]]),
        ]

        for ish in h:
            for jsh in he:
                overlap = overlap_cgto(ish, jsh, r2, vec, 100.0)
                self.assertTrue(
                    torch.allclose(
                        overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
                    ),
                    msg=f"Overlap does not match:\n {overlap}",
                )

    def test_overlap_s_cl(self):
        """
        Compare against reference calculated with tblite-int S Cl 0,0,2.1 --method gfn1 --bohr
        """
        from xtbml.param.gfn1 import GFN1_XTB as par
        from xtbml.xtb.calculator import Basis

        basis = Basis(["S", "Cl"], par)
        s = basis.cgto.get("S")
        cl = basis.cgto.get("Cl")

        vec = torch.tensor([0.0, 0.0, 2.1])
        r2 = vec.dot(vec)

        ref = [
            torch.tensor([[4.21677786e-01]]),
            torch.tensor([[0.0], [0.0], [-5.53353521e-01]]),
            torch.tensor([[+3.57843134e-01], [0.0], [0.0], [0.0], [0.0]]),
            torch.tensor([[0.0, 0.0, 5.71415151e-01]]),
            torch.diag(
                torch.tensor(
                    [4.48181566e-01, 4.48181566e-01, -2.31133305e-01],
                ),
            ),
            torch.tensor(
                [
                    [0.0, 0.0, -0.07887177],
                    [-0.49594098, 0.0, 0.0],
                    [0.0, -0.49594098, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ),
            torch.tensor([[0.40834189, 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor(
                [
                    [0.0, 0.49152805, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.49152805, 0.0, 0.0],
                    [0.05231871, 0.0, 0.0, 0.0, 0.0],
                ],
            ),
            torch.diag(
                torch.tensor(
                    [-0.03689737, -0.29504313, -0.29504313, 0.35847499, 0.35847499]
                )
            ),
        ]

        for ish in s:
            for jsh in cl:
                overlap = overlap_cgto(ish, jsh, r2, vec, 100.0)
                self.assertTrue(
                    torch.allclose(
                        overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
                    ),
                    msg=f"Overlap does not match:\n {overlap}",
                )
