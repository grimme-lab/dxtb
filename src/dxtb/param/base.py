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

from pydantic import BaseModel
from pathlib import Path
import torch

from .charge import Charge
from .dispersion import Dispersion
from .element import Element
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import Repulsion
from .thirdorder import ThirdOrder
from ..utils.utils import (
    rgetattr,
    rsetattr,
    get_attribute_name_key,
    get_all_entries_from_dict,
)


class Param(BaseModel):
    """
    Complete self-contained representation of an extended tight-binding model.
    """

    meta: Meta | None
    """Descriptive data on the model"""
    element: dict[str, Element]
    """Element specific parameter records"""
    hamiltonian: Hamiltonian
    """Definition of the Hamiltonian, always required"""
    dispersion: Dispersion | None
    """Definition of the dispersion correction"""
    repulsion: Repulsion | None
    """Definition of the repulsion contribution"""
    charge: Charge | None
    """Definition of the isotropic second-order charge interactions"""
    multipole: dict | None = None
    """Definition of the anisotropic second-order multipolar interactions (not implemented)"""
    halogen: Halogen | None = None
    """Definition of the halogen bonding correction (not implemented)"""
    thirdorder: ThirdOrder | None
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
        elif "." in key:
            key, attr = key.split(".")
            return rgetattr(rgetattr(self, name)[key], attr)
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
        elif "." in key:
            key, attr = key.split(".")
            d = rgetattr(self, name)
            rsetattr(d[key], attr, value)
            rsetattr(self, name, d)
        else:
            d = rgetattr(self, name)
            d[key] = value
            rsetattr(self, name, d)

    def to_toml(
        self, path: str | Path = "gfn1-xtb.toml", overwrite: bool = False
    ) -> None:
        """
        Export parametrization to TOML file.

        Parameters
        ----------
        path : str | Path, optional
            Path for output file (default: "gfn1-xtb.toml")
        overwrite: bool
            Whether to overwrite the file if it already exists.
        """

        # pylint: disable=import-outside-toplevel
        import toml

        path = Path(path)
        if path.is_file() is True and overwrite is False:
            raise FileExistsError(f"File '{path}' already exists.")

        with open(path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(self.dict()))

    def get_all_param_names(self) -> list[str]:
        """Obtain all parameter names contained in Parametrisation object.

        Returns
        -------
        list[str]
            List with all parameter names.
        """

        def obj_to_list_of_strings(
            obj, list_str=[], running_str=[]
        ) -> tuple[list[str], list[str]]:
            """Recursively iterate through nested object and concat attributes to strings."""

            for item in sorted(obj.__dict__):
                running_str.append(str(item))

                if hasattr(obj.__dict__[item], "__dict__"):
                    _, running_str = obj_to_list_of_strings(
                        obj.__dict__[item], list_str, running_str
                    )

                if len(running_str) > 1:
                    list_str.append(".".join(running_str))
                running_str = running_str[:-1]

            return list_str, running_str

        str_list, _ = obj_to_list_of_strings(self, [])

        # expand dictionary and lists
        li = []
        for s in str_list:
            ent = get_all_entries_from_dict(self, s)
            if isinstance(ent, list):
                li.extend(ent)
            else:
                li.append(ent)

        # add element variables
        for k, v in self.element.items():
            for k2 in vars(v).keys():
                # NOTE: ignore non-numerical shell specification
                if k2 != "shells":
                    li.append(f"element['{k}'].{k2}")

        # remove non-numeric and "duplicated" entries
        li = [
            s for s in li if type(self.get_param(s)) in [int, float, list, torch.tensor]
        ]
        return li
