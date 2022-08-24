# This file is part of xtbml.

"""
Definition of the full parametrization data for the extended tight-binding methods.

The dataclass can represent a complete parametrization file produced by the `tblite`_
library, however it only stores the raw data rather than the full representation.

The parametrization of a calculator with the model data must account for missing
transformations, like extracting the principal quantum numbers from the shells.
The respective checks are therefore deferred to the instantiation of the calculator,
while a deserialized model in `tblite`_ is already verified at this stage.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel

from .dispersion import Dispersion
from .charge import Charge
from .element import Element
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import Repulsion
from .thirdorder import ThirdOrder
from ..typing import Tensor
from ..utils.utils import rgetattr, rsetattr, get_attribute_name_key


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.
    """

    meta: Optional[Meta]
    """Descriptive data on the model"""
    element: Dict[str, Element]
    """Element specific parameter records"""
    hamiltonian: Hamiltonian
    """Definition of the Hamiltonian, always required"""
    dispersion: Optional[Dispersion]
    """Definition of the dispersion correction"""
    repulsion: Optional[Repulsion]
    """Definition of the repulsion contribution"""
    charge: Optional[Charge]
    """Definition of the isotropic second-order charge interactions"""
    multipole: Optional[dict] = None
    """Definition of the anisotropic second-order multipolar interactions (not implemented)"""
    halogen: Optional[Halogen] = None
    """Definition of the halogen bonding correction (not implemented)"""
    thirdorder: Optional[ThirdOrder]
    """Definition of the isotropic third-order charge interactions"""

    def get_param(
        self,
        name: str,
    ) -> Any:
        """Get single parameter based on string. Works on nested attributes including dictionary entries.

        Parameters
        ----------
        name : str
            Identifier for specific (nested) attribute.

        Returns
        -------
        Any
            Return value from parametrisation.
        """
        name, key = get_attribute_name_key(name)

        if key is None:
            return rgetattr(self, name)
        else:
            return rgetattr(self, name)[key]

    def set_param(self, name: str, value: Any):
        """Set value of single parameter.

        Parameters
        ----------
        name : str
            Identifier for specific (nested) attribute.
        value : Any
            Value to be assigned.
        """
        name, key = get_attribute_name_key(name)

        if key is None:
            rsetattr(self, name, value)
        else:
            d = rgetattr(self, name)
            d[key] = value
            rsetattr(self, name, d)
