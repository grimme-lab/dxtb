"""
Container for interactions.
"""
from __future__ import annotations

import torch

from .._types import Any, Literal, Slicers, Tensor, TensorOrTensors, overload
from ..basis import IndexHelper
from ..coulomb import secondorder, thirdorder
from .base import Interaction
from .container import Charges, Potential
from .external import field as efield

__all__ = ["InteractionList"]


class InteractionList(Interaction):
    """
    List of interactions.
    """

    class Cache(dict):
        """
        List of interaction caches.
        """

        __slots__ = ()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            """
            Cull all interaction caches.

            Parameters
            ----------
            conv : Tensor
                Mask of converged systems.
            """
            for cache in self.values():
                cache.cull(conv, slicers)

        def restore(self) -> None:
            """
            Restore all interaction caches.
            """
            for cache in self.values():
                cache.restore()

    def __init__(self, *interactions: Interaction | None) -> None:
        # FIXME: Defaults?
        super().__init__(torch.device("cpu"), torch.float)
        self.interactions = [
            interaction for interaction in interactions if interaction is not None
        ]

    @property
    def labels(self) -> list[str]:
        return [interaction.label for interaction in self.interactions]

    @overload
    def get_interaction(self, name: Literal["ElectricField"]) -> efield.ElectricField:
        ...

    @overload
    def get_interaction(self, name: Literal["ES2"]) -> secondorder.ES2:
        ...

    @overload
    def get_interaction(self, name: Literal["ES3"]) -> thirdorder.ES3:
        ...

    def get_interaction(self, name: str) -> Interaction:
        """
        Obtain an interaction from the list of interactions by its class name.

        Parameters
        ----------
        name : str
            Name of the interaction.

        Returns
        -------
        Interaction
            Instance of the interaction as present in the `InteractionList`.

        Raises
        ------
        ValueError
            Unknown interaction name given or interaction is not in the list.
        """
        for interaction in self.interactions:
            if name == interaction.label:
                return interaction

        raise ValueError(f"The interaction named '{name}' is not in the list.")

    def update(self, name: str, **kwargs: Any) -> None:
        """
        Update the attributes of an interaction object within the list.

        This method iterates through the interactions in the list, finds the
        one with the matching label, and updates its attributes based on the
        provided arguments.

        Parameters
        ----------
        name : str
            The label of the interaction object to be updated.
        **kwargs : dict
            Keyword arguments containing the attributes and their new values to
            be updated in the interaction object.

        Raises
        ------
        ValueError
            If no interaction with the given label is found in the list.

        Examples
        --------
        >>> from dxtb.interaction import InteractionList
        >>> from dxtb.interaction.external import field as efield
        >>>
        >>> field_vector = torch.tensor([0.0, 0.0, 0.0])
        >>> new_field_vector = torch.tensor([1.0, 0.0, 0.0])
        >>> ef = efield.new_efield(field_vector)
        >>> ilist = InteractionList(ef)
        >>> ilist.update(efield.LABEL_EFIELD, field=new_field_vector)
        """
        for interaction in self.interactions:
            if name == interaction.label:
                interaction.update(**kwargs)
                return

        raise ValueError(f"The interaction named '{name}' is not in the list.")

    def update_efield(
        self, *, field: Tensor | None = None, field_grad: Tensor | None = None
    ) -> None:
        return self.update(
            efield.LABEL_EFIELD,
            field=field,
            field_grad=field_grad,
        )

    def update_es2(self, **kwargs: Any) -> None:
        return self.update(secondorder.LABEL_ES2, **kwargs)

    def update_es3(self, **kwargs: Any) -> None:
        return self.update(thirdorder.LABEL_ES3, **kwargs)

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> InteractionList.Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionList.Cache
            Restart data for the interactions.
        """
        cache = self.Cache()
        cache.update(
            **{
                interaction.label: interaction.get_cache(
                    numbers=numbers, positions=positions, ihelp=ihelp
                )
                for interaction in self.interactions
            }
        )
        return cache

    def get_potential(
        self, charges: Charges, cache: InteractionList.Cache, ihelp: IndexHelper
    ) -> Potential:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionList.Cache
            Restart data for the interactions.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        # create empty potential
        pot = Potential(
            torch.zeros_like(charges.mono),
            dipole=None,
            quad=None,
            batched=ihelp.batched,
        )

        # exit with empty potential if no interactions present
        if len(self.interactions) <= 0:
            return pot

        # add up potentials from all interactions
        for interaction in self.interactions:
            p = interaction.get_potential(charges, cache[interaction.label], ihelp)
            pot += p

        return pot

    def get_energy_as_dict(
        self, charges: Charges, cache: Cache, ihelp: IndexHelper
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """
        if len(self.interactions) <= 0:
            return {"none": torch.zeros_like(charges.mono)}

        return {
            interaction.label: interaction.get_energy(
                charges, cache[interaction.label], ihelp
            )
            for interaction in self.interactions
        }

    def get_energy(
        self, charges: Charges | Tensor, cache: Cache, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges | Tensor
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-resolved energy vector for orbital partial charges.
        """
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        if len(self.interactions) <= 0:
            return ihelp.reduce_orbital_to_atom(torch.zeros_like(charges.mono))

        return torch.stack(
            [
                interaction.get_energy(charges, cache[interaction.label], ihelp)
                for interaction in self.interactions
            ]
        ).sum(dim=0)

    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionList.Cache,
        ihelp: IndexHelper,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculate gradient for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        cache : InteractionList.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient of all interactions.
        """
        if len(self.interactions) <= 0:
            return torch.zeros_like(positions)

        return torch.stack(
            [
                interaction.get_gradient(
                    charges,
                    positions,
                    cache[interaction.label],
                    ihelp,
                    grad_outputs=grad_outputs,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
                for interaction in self.interactions
            ]
        ).sum(dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels})"
