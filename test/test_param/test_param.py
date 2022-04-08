# This file is part of xtbml.

from typing import List
import tomli as toml
import torch
from unittest import TestCase

from xtbml.data.covrad import to_number
from xtbml.exlibs.tbmalt import Geometry
import xtbml.param


class TestParam(TestCase):
    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def test_builtin_gfn1(self):
        from xtbml.param.gfn1 import GFN1_XTB as par

        self.assertTrue(par.meta.name == "GFN1-xTB")
        self.assertTrue(par.meta.version == 1)

        self.assertTrue(par.dispersion is not None)
        self.assertTrue(par.repulsion is not None)
        self.assertTrue(par.charge is not None)
        self.assertTrue(par.multipole is None)
        self.assertTrue(par.halogen is not None)
        self.assertTrue(par.thirdorder is not None)

        self.assertTrue(par.hamiltonian.xtb.cn == "exp")
        self.assertTrue("sp" in par.hamiltonian.xtb.shell)
        self.assertTrue(par.hamiltonian.xtb.wexp == 0.0)
        self.assertTrue(par.hamiltonian.xtb.kpol == 2.85)
        self.assertTrue(par.hamiltonian.xtb.enscale == -7.0e-3)

        self.assertTrue("Te" in par.element)

    def test_param_minimal(self):
        data = """
        [hamiltonian.xtb]
        wexp = 5.0000000000000000E-01
        kpol = 2.0000000000000000E+00
        enscale = 2.0000000000000000E-02
        cn = "gfn"
        shell = {ss=1.85, pp=2.23, dd=2.23, sd=2.00, pd=2}
        [element.H]
        shells = [ "1s" ]
        levels = [ -1.0707210999999999E+01 ]
        slater = [ 1.2300000000000000E+00 ]
        ngauss = [ 3 ]
        refocc = [ 1.0000000000000000E+00 ]
        shpoly = [ -9.5361800000000000E-03 ]
        kcn = [ -5.0000000000000003E-02 ]
        gam = 4.0577099999999999E-01
        lgam = [ 1.0000000000000000E+00 ]
        gam3 = 8.0000000000000016E-02
        zeff = 1.1053880000000000E+00
        arep = 2.2137169999999999E+00
        xbond = 0.0000000000000000E+00
        dkernel = 5.5638889999999996E-02
        qkernel = 2.7430999999999999E-04
        mprad = 1.3999999999999999E+00
        mpvcn = 1.0000000000000000E+00
        en = 2.2000000000000002E+00
        [element.C]
        shells = [ "2s", "2p" ]
        levels = [ -1.3970922000000002E+01, -1.0063292000000001E+01 ]
        slater = [ 2.0964320000000001E+00, 1.8000000000000000E+00 ]
        ngauss = [ 4, 4 ]
        refocc = [ 1.0000000000000000E+00, 3.0000000000000000E+00 ]
        shpoly = [ -2.2943210000000002E-02, -2.7110200000000002E-03 ]
        kcn = [ -1.0214400000000000E-02, 1.6165700000000002E-02 ]
        gam = 5.3801500000000002E-01
        lgam = [ 1.0000000000000000E+00, 1.1056357999999999E+00 ]
        gam3 = 1.5000000000000002E-01
        zeff = 4.2310780000000001E+00
        arep = 1.2476550000000000E+00
        xbond = 0.0000000000000000E+00
        dkernel = -4.1167399999999998E-03
        qkernel = 2.1358300000000000E-03
        mprad = 3.0000000000000000E+00
        mpvcn = 3.0000000000000000E+00
        en = 2.5499999999999998E+00
        """

        par = xtbml.param.Param(**toml.loads(data))

        self.assertTrue("H" in par.element)
        self.assertTrue("C" in par.element)

    def test_param_calculator(self):
        from xtbml.xtb.calculator import Calculator
        from xtbml.param.gfn1 import GFN1_XTB as par

        atomic_numbers = symbol2number(["H", "C"])
        dummy_coords = torch.zeros(3)
        mol = Geometry(atomic_numbers, dummy_coords)

        calc = Calculator(mol, par)

        self.assertTrue("H" in calc.basis.cgto)
        self.assertTrue("C" in calc.basis.cgto)

        self.assertTrue("H" in calc.hamiltonian.refocc)
        self.assertTrue("C" in calc.hamiltonian.refocc)

        self.assertTrue(sum(calc.hamiltonian.refocc.get("H")) == 1)
        self.assertTrue(sum(calc.hamiltonian.refocc.get("C")) == 4)


def symbol2number(symList: List[str]) -> torch.Tensor:
    return torch.flatten(torch.tensor([to_number(s) for s in symList]))
