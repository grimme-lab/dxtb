

from abc import ABC
import torch


class Tb_H0spec(ABC):
    """Specification of the effective Hamiltonian.
    An instance of the specification is consumed by the constructor of the
    Hamiltonian to generate the respective entries in the derived type."""
   

    def get_hscale(self, mol, bas, hscale):
        """ Generator for the enhancement factor to for scaling Hamiltonian elements """
        """import :: tb_h0spec, structure_type, basis_type, wp
        # Instance of the Hamiltonian specification
        class(tb_h0spec), intent(in) :: self
        # Molecular structure data
        type(structure_type), intent(in) :: mol
        # Basis set information
        type(basis_type), intent(in) :: bas"""
        # Scaling parameters for the Hamiltonian elements
        hscale = torch.tensor([0, 0, 0, 0])
        return hscale

# TODO: if needed, goon with implementing interface