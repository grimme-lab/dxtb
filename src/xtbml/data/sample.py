from typing import Dict, List
from pydantic import BaseModel
import torch
from torch import Tensor
from ase.build import molecule

from xtbml.adjlist import AdjacencyList
from xtbml.basis.type import get_cutoff
from xtbml.data.covrad import get_covalent_rad
from xtbml.ncoord.ncoord import get_coordination_number
from xtbml.xtb.calculator import Calculator
from xtbml.constants import FLOAT32, FLOAT64, UINT8
from xtbml.param.gfn1 import GFN1_XTB as gfn1_par

from xtbml.exlibs.tbmalt import Geometry
from xtbml.exlibs.tbmalt.batch import unpack


class Sample(BaseModel):
    """Representation for single sample information."""

    # supported features
    uid: str
    """Unique identifier for sample"""
    positions: Tensor
    """Atomic positions"""
    atomic_numbers: Tensor
    """Atomic numbers"""
    egfn1: Tensor
    """Energy calculated by GFN1-xtb"""
    overlap: Tensor
    """Overlap matrix"""
    hamiltonian: Tensor
    """Hamiltonian matrix"""
    # ...
    # TODO: add further QM features

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    def from_json() -> "Sample":
        """
        Create sample from json.
        """
        raise NotImplementedError

    def to_json(self) -> Dict:
        """
        Convert sample to json.
        """
        raise NotImplementedError

    def calc_singlepoint(data: Geometry) -> List[Dict]:
        """Calculate QM features based on classicals input, such as geometry."""

        # DUMMY INPUT DATA
        data = Geometry.from_ase_atoms(
            # molecule("CH4")  # single
            # [molecule("CO2"), molecule("H2O")]  # same size batch
            [molecule("CH4"), molecule("C2H4"), molecule("H2O")]  # batch
        )

        # CALC FEATURES

        # parametrisation
        par = gfn1_par
        cn_cutoff = 30.0

        features = []

        for sample in data:

            # NOTE: singlepoint calculation so far only working with no padding
            # quick and dirty hack to get rid of padding
            s = Geometry(
                atomic_numbers=unpack(sample.atomic_numbers.unsqueeze(0))[0],
                positions=unpack(sample.positions.unsqueeze(0))[0],
                charges=sample.charges,
                unpaired_e=sample.unpaired_e,
            )

            # check underlying data consistent
            assert (
                sample.atomic_numbers.storage().data_ptr()
                == s.atomic_numbers.storage().data_ptr()
            )

            # single point calculation
            calc = Calculator(s, par)

            # empty translation (no PBC yet)
            trans = torch.zeros([3, 1])
            rcov = get_covalent_rad(sample.chemical_symbols)
            cn = get_coordination_number(sample, trans, cn_cutoff, rcov)
            adj = AdjacencyList(sample, trans, get_cutoff(calc.basis))

            # hamiltonian
            h, overlap_int = calc.hamiltonian.build(calc.basis, adj, cn)

            features.append(
                {
                    "hamiltonian": h,
                    "overlap": overlap_int,
                    "coordination_number": cn,
                    "adjacency_list": adj,
                }
            )

        return features
