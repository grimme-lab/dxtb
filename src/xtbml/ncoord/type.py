from abc import ABC, abstractmethod


class NcoordType(ABC):
    @abstractmethod
    def get_cn(self, mol, cn, dcndr, dcndL):
        """# Coordination number container
        class(ncoord_type), intent(in) :: self
        # Molecular structure data
        type(structure_type), intent(in) :: mol
        # Error function coordination number.
        real(wp), intent(out) :: cn(:)
        # Derivative of the CN with respect to the Cartesian coordinates.
        real(wp), intent(out), optional :: dcndr(:, :, :)
        # Derivative of the CN with respect to strain deformations.
        real(wp), intent(out), optional :: dcndL(:, :, :)"""
        return cn, dcndr, dcndL
