from ase.build import molecule
import torch
from unittest import TestCase
from typing import Generator, List

from xtbml.exlibs.tbmalt import Geometry


def generator_test(
    generator: Generator[torch.tensor, None, None],
    expected_values: List[torch.tensor],
) -> bool:
    """Helper method to test generators for correct output.

    Args:
        generator (Generator[torch.tensor, None, None]): Generator yielding torch.tensors.
        expected_values (List[torch.tensor]): Sorted list of expected results.

    Returns:
        bool: True if all elements are equal.
    """

    return all(
        [torch.equal(value, expected_values[i]) for i, value in enumerate(generator)]
    )


class Test_Geometry(TestCase):
    """Testing the construction and properties of Geometry objects."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def test_from_ase(self) -> None:
        """Test creation of single geometry from ase."""
        geometry = Geometry.from_ase_atoms([molecule("CH4")])
        atn = torch.tensor([[6, 1, 1, 1, 1]])
        pos = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [1.188861, 1.188861, 1.188861],
                    [-1.188861, -1.188861, 1.188861],
                    [1.188861, -1.188861, -1.188861],
                    [-1.188861, 1.188861, -1.188861],
                ]
            ]
        )

        self.assertTrue(
            torch.equal(geometry.atomic_numbers, atn),
            msg=f"Wrong atomic numbers",
        )
        self.assertTrue(
            torch.allclose(geometry.positions, pos, atol=1e-8),
            msg=f"Wrong geometry",
        )

    def test_from_ase_batch(self) -> None:
        """Test creation of batch geometry from ase."""
        geometry = Geometry.from_ase_atoms([molecule("CH4"), molecule("C2H4")])
        atn = torch.tensor([[6, 1, 1, 1, 1, 0], [6, 6, 1, 1, 1, 1]])
        pos = torch.tensor(
            [
                [
                    [0.00000000, 0.00000000, 0.00000000],
                    [1.18886077, 1.18886077, 1.18886077],
                    [-1.18886077, -1.18886077, 1.18886077],
                    [1.18886077, -1.18886077, -1.18886077],
                    [-1.18886077, 1.18886077, -1.18886077],
                    [0.00000000, 0.00000000, 0.00000000],
                ],
                [
                    [0.00000000, 0.00000000, 1.26135433],
                    [0.00000000, 0.00000000, -1.26135433],
                    [0.00000000, 1.74389970, 2.33890438],
                    [0.00000000, -1.74389970, 2.33890438],
                    [0.00000000, 1.74389970, -2.33890438],
                    [0.00000000, -1.74389970, -2.33890438],
                ],
            ]
        )

        self.assertTrue(
            torch.equal(geometry.atomic_numbers, atn),
            msg=f"Wrong atomic numbers",
        )
        self.assertTrue(
            torch.allclose(geometry.positions, pos, atol=1e-8),
            msg=f"Wrong geometry",
        )

    def test_generate_interactions(self) -> None:
        """Test generation of interactions for batch geometry from ase."""

        geometry = Geometry.from_ase_atoms([molecule("CH4"), molecule("C2H4")])

        expected_values = [
            torch.tensor([1, 1]),
            torch.tensor([1, 6]),
            torch.tensor([6, 6]),
        ]

        self.assertTrue(
            generator_test(
                geometry.generate_interactions(unique=True), expected_values
            ),
            msg=f"Wrong interactions",
        )

    def test_setting_charges(self) -> None:
        """Test for checking the setting of charge property - in batch and single geometry." """

        geometry = Geometry.from_ase_atoms(
            [molecule("CH4"), molecule("CH4"), molecule("C2H4")]
        )

        chg_intial = torch.tensor([0.0, 0.0, 0.0])
        chg = torch.tensor([1.0, 2.0, 3.0])

        self.assertTrue(
            torch.equal(geometry.charges, chg_intial),
            msg=f"Wrong inital charges",
        )
        # changing geometry[i]._charges = x does not do anything
        geometry[1]._charges = torch.tensor([4.0])
        self.assertTrue(
            torch.equal(geometry.charges, chg_intial),
            msg=f"Wrong changed charges",
        )
        self.assertTrue(
            torch.equal(geometry[1].charges, chg_intial[1]),
            msg=f"Wrong individual inital charges",
        )

        # manually change charges and uhf
        geometry._charges = chg
        self.assertTrue(
            torch.equal(geometry.charges, chg),
            msg=f"Wrong updated charges",
        )
        self.assertTrue(
            torch.equal(geometry[1].charges, chg[1]),
            msg=f"Wrong individual charges",
        )

    def test_setting_uhf(self) -> None:
        """Test for checking the setting of unpaired electron property - in batch and single geometry." """

        geometry = Geometry.from_ase_atoms(
            [molecule("CH4"), molecule("CH4"), molecule("C2H4")]
        )

        uhf_intial = torch.tensor([0.0, 0.0, 0.0])
        uhf = torch.tensor([4.0, 5.0, 6.0])

        self.assertTrue(
            torch.equal(geometry.unpaired_e, uhf_intial),
            msg=f"Wrong inital unpaired electrons",
        )
        # changing geometry[i]._unpaired_e = x does not do anything
        geometry[1]._unpaired_e = torch.tensor([4.0])
        self.assertTrue(
            torch.equal(geometry.unpaired_e, uhf_intial),
            msg=f"Wrong changed unpaired electrons",
        )
        self.assertTrue(
            torch.equal(geometry[1].unpaired_e, uhf_intial[1]),
            msg=f"Wrong individual inital unpaired electrons",
        )

        # manually change unpaired_e and uhf
        geometry._unpaired_e = uhf
        self.assertTrue(
            torch.equal(geometry.unpaired_e, uhf),
            msg=f"Wrong updated unpaired electrons",
        )
        self.assertTrue(
            torch.equal(geometry[1].unpaired_e, uhf[1]),
            msg=f"Wrong individual unpaired electrons",
        )
