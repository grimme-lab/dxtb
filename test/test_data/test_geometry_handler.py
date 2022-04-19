from ase.build import molecule
from unittest import TestCase
from typing import List
from pathlib import Path

from xtbml.exlibs.tbmalt import Geometry
from xtbml.data.geometry_handler import Geometry_Handler as gh


class Test_Geometry_Handler(TestCase):
    """Testing handling of batched Geometry objects."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def test_geometry_filter(self) -> None:
        """Test filtering batched Geometry objects with Geometry_Handler."""

        # should actually contain file paths (doesn't affect the testing though)
        sample_list = ["C2H4", "H2O", "NH3", "H2", "CO", "NaCl"]

        geometry = Geometry.from_ase_atoms(
            [
                molecule("C2H4"),
                molecule("H2O"),
                molecule("NH3"),
                molecule("H2"),
                molecule("CO"),
                molecule("NaCl"),
            ]
        )

        def filter_include_C2H4(file_list: List[str]) -> List[int]:
            mask = []
            for i, s in enumerate(file_list):
                filename = Path(s).stem
                if "C2H4" in filename:
                    mask.append(i)
            return mask

        new_geometry, new_sample_list = gh.filter_geometry_by_filelist(
            geometry, sample_list, filter_include_C2H4
        )

        self.assertTrue(
            new_sample_list == ["C2H4"],
            msg=f"Incorrect sample list",
        )

        self.assertTrue(
            new_geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("C2H4"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )

    def test_geometry_filelist_filter(self) -> None:
        """Test filtering batched Geometry objects with Geometry_Handler by filelist."""

        # should actually contain file paths (doesn't affect the testing though)
        sample_list = ["charged+", "neutral", "neutral", "charged-"]

        geometry = Geometry.from_ase_atoms(
            [
                molecule("C2H4"),
                molecule("H2O"),
                molecule("NH3"),
                molecule("H2"),
            ]
        )

        new_geometry, new_sample_list = gh.filter_geometry_by_filelist(
            geometry, sample_list, gh.filter_only_charged
        )

        self.assertTrue(
            new_sample_list == ["charged+", "charged-"],
            msg=f"Incorrect sample list",
        )

        self.assertTrue(
            new_geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("C2H4"),
                    molecule("H2"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )

    def test_geometry_removal(self) -> None:
        """Test removing single Geometries from batched Geometry."""

        geometry = Geometry.from_ase_atoms(
            [
                molecule("CO"),
                molecule("H2O"),
                molecule("NH3"),
                molecule("H2"),
                molecule("C2H4"),
                molecule("NaCl"),
            ]
        )

        geometry = gh.remove_sample_from_geometry(geometry, 1)
        # NOTE: for this comparison, leave largest molecule in structure for identical padding
        self.assertTrue(
            geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("CO"),
                    molecule("NH3"),
                    molecule("H2"),
                    molecule("C2H4"),
                    molecule("NaCl"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )
        geometry = gh.remove_sample_from_geometry(geometry, 2)
        self.assertTrue(
            geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("CO"),
                    molecule("NH3"),
                    molecule("C2H4"),
                    molecule("NaCl"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )

        geometry = gh.remove_sample_from_geometry(geometry, -1)
        self.assertTrue(
            geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("CO"),
                    molecule("NH3"),
                    molecule("C2H4"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )

        self.assertListEqual(
            list(geometry.atomic_numbers.shape),
            [3, 6],
            msg="Inconsistent padding",
        )

        geometry = gh.remove_sample_from_geometry(geometry, -2)
        self.assertTrue(
            geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("CO"),
                    molecule("C2H4"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )

        self.assertRaises(IndexError, gh.remove_sample_from_geometry, geometry, -4)

        geometry = gh.remove_sample_from_geometry(geometry, 0)
        self.assertTrue(
            geometry
            == Geometry.from_ase_atoms(
                [
                    molecule("C2H4"),
                ]
            ),
            msg=f"Incorrect geometry object",
        )
