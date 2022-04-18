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

from typing import Dict, Optional
from pydantic import BaseModel


from .dispersion import Dispersion
from .element import Element
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import Repulsion


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.
    """

    meta: Meta
    """Descriptive data on the model"""
    element: Dict[str, Element]
    """Element specific parameter records"""
    hamiltonian: Hamiltonian
    """Definition of the Hamiltonian, always required"""
    dispersion: Dispersion
    """Definition of the dispersion correction (not implemented)"""
    repulsion: Repulsion
    """Definition of the repulsion contribution"""
    charge: Optional[dict] = None
    """Definition of the isotropic second-order charge interactions (not implemented)"""
    multipole: Optional[dict] = None
    """Definition of the anisotropic second-order multipolar interactions (not implemented)"""
    halogen: Optional[Halogen] = None
    """Definition of the halogen bonding correction (not implemented)"""
    thirdorder: Optional[dict] = None
    """Definition of the isotropic third-order charge interactions (not implemented)"""
