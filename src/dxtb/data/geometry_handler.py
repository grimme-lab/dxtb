from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch

from ..exlibs.tbmalt import Geometry


class Geometry_Handler:
    """Class to operate on (batched) Geometry objects."""

    def remove_sample_from_geometry(
        geometry: Geometry, idx_to_remove: int
    ) -> Geometry | None:
        """Crude implementation of removing single geometry from batch geometry object.

        Args:
            geometry (Geometry): Batch geometry.
            idx_to_remove (int): Index of single geometry within batch geometry.

        Returns:
            Geometry: Pruned batch geometry.
        """
        # last sample in geometry
        if len(geometry) == 1:
            return None

        # translate relative indices
        if idx_to_remove < 0:
            idx_to_remove = len(geometry) + idx_to_remove
            if idx_to_remove < 0:
                raise IndexError(
                    "Cannot remove geometry from batch. Batch smaller than chosen index."
                )

        # add geometries together (except for one)
        idx_start = 1 if idx_to_remove == 0 else 0
        new_geometry = geometry[idx_start]
        for i in range(idx_start + 1, len(geometry)):
            if i == idx_to_remove:
                continue
            new_geometry = new_geometry + geometry[i]

        if len(new_geometry) == 1:
            # set shape of properties to batch size 1
            new_geometry = Geometry(
                atomic_numbers=torch.unsqueeze(new_geometry.atomic_numbers, 0),
                positions=torch.unsqueeze(new_geometry.positions, 0),
                charges=torch.unsqueeze(new_geometry.charges, 0),
                unpaired_e=torch.unsqueeze(new_geometry.unpaired_e, 0),
            )

        return new_geometry

    def select(geometry: Geometry, selector_fn: Callable[[Geometry], bool]) -> Geometry:
        """Generic select method. Obtain single function operating on geometry slices.

        Args:
            geometry (Geometry): Batch geometry holding all samples.
            selector_fn (function): Masking function to select specific single geometries in batch.

        Returns:
            Geometry: Batch geometry holding samples that fulfill selector criteria.
        """
        remove_list = []
        for i, g in enumerate(geometry):
            if not selector_fn(g):
                remove_list.append(i)

        for i in reversed(remove_list):
            geometry = Geometry_Handler.remove_sample_from_geometry(geometry, i)
        return geometry

    def selector_HCNO(geometry: Geometry) -> bool:
        """Selector function, checking for HCNO elements only.

        Args:
            geometry (Geometry): Single geometry to be evaluated.

        Returns:
            bool: Flag returning True if only HCNO elements present, else False.
        """
        # allowed elements (plus padding constant)
        hcno = [1, 6, 7, 8] + [0]
        return all([e in hcno for e in set(geometry.atomic_numbers.tolist())])

    def filter_geometry_by_filelist(
        geometry: Geometry,
        file_list: list[str],
        filter_rule: Callable[[list[str]], list[int]],
    ) -> tuple[Geometry, list[str]] | tuple[None, list[str]]:

        filter = filter_rule(file_list)
        mask = [i for i in range(len(geometry)) if i not in filter]
        file_list = [f for i, f in enumerate(file_list) if i in filter]

        for i in reversed(mask):
            geometry = Geometry_Handler.remove_sample_from_geometry(geometry, i)

        return geometry, file_list

    def filter_only_charged(file_list: list[str]) -> list[int]:
        """Filter for samples with charged (i.e. not neutral) compounds.

        Args:
            file_list (List[str]): List containing all sample (file)names.

        Returns:
            List[int]: List containing indices of remaining samples.
        """
        mask = []
        for i, s in enumerate(file_list):
            filename = Path(s).stem
            if "+" in filename or "-" in filename:
                mask.append(i)
        return mask
