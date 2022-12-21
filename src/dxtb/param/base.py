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
from __future__ import annotations

from pydantic import BaseModel

from .._types import Optional
from .charge import Charge
from .dispersion import Dispersion
from .element import Elements
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import Repulsion
from .thirdorder import ThirdOrder


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.
    """

    meta: Optional[Meta]
    """Descriptive data on the model."""
    element: Elements
    """Element specific parameter records."""
    hamiltonian: Hamiltonian
    """Definition of the Hamiltonian, always required."""
    dispersion: Optional[Dispersion] = None
    """Definition of the dispersion correction."""
    repulsion: Optional[Repulsion] = None
    """Definition of the repulsion contribution."""
    charge: Optional[Charge] = None
    """Definition of the isotropic second-order charge interactions."""
    multipole: Optional[dict] = None
    """Definition of the anisotropic second-order multipolar interactions."""
    halogen: Optional[Halogen] = None
    """Definition of the halogen bonding correction."""
    thirdorder: Optional[ThirdOrder] = None
    """Definition of the isotropic third-order charge interactions."""
