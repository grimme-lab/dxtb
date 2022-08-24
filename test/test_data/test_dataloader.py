from ase.build import molecule
from unittest import TestCase
import torch
from os.path import isdir

from xtbml.exlibs.tbmalt import Geometry
from xtbml.data.datareader import Datareader


class Test_Dataloader(TestCase):
    """Testing the loading of batch Geometries objects into dataloader."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def setUp(self):
        # root directory of GFN2-fitdata
        self.path = "../data/gfn2-fitdata/data/"
        self.has_gfn2 = isdir(self.path)

        # dtype for charges and unpaired electrons
        self.dtype_charges = torch.int8
        self.dtype_uhf = torch.uint8

    def test_dataloader_from_ase(self) -> None:
        """Test constructing a dataloader from Geometry object and config."""

        geometry = Geometry.from_ase_atoms(
            [molecule("CH4"), molecule("H2O"), molecule("C2H4")]
        )

        # setup config for pytorch dataloader class
        cfg = {
            "batch_size": 2,
            "shuffle": False,
            "sampler": None,
            "batch_sampler": None,
            "num_workers": 0,
            # "collate_fn" : None, #-> default given in Datareader.get_dataloader
            "pin_memory": False,
            "drop_last": False,
            "timeout": 0,
            "worker_init_fn": None,
            "multiprocessing_context": None,
            "generator": None,
            "prefetch_factor": 2,
            "persistent_workers": False,
        }

        # use dataloader to iterate over batched geometries
        dataloader = Datareader.get_dataloader(geometry, cfg)

        # especially testing that collating Geometry objects works
        for i, sample in enumerate(dataloader):

            self.assertTrue(
                isinstance(sample, Geometry),
                msg="Each batch must be Geometry object",
            )
            self.assertTrue(
                len(sample) in [1, 2],  # incl. last batch
                msg="Wrong batch_size",
            )

            try:
                self.assertListEqual(
                    list(sample.atomic_numbers.shape),
                    [2, 6],
                    msg="Inconsistent padding",
                )
            except AssertionError:  # incl. last batch
                self.assertListEqual(
                    list(sample.atomic_numbers.shape),
                    [6],
                    msg="Inconsistent padding",
                )
            self.assertTrue(
                torch.equal(
                    sample.charges, torch.tensor(0, dtype=self.dtype_charges)
                )  # incl. last batch
                or torch.equal(
                    sample.charges, torch.tensor([0, 0], dtype=self.dtype_charges)
                ),
                msg="Inconsistent charge for molecules generated from ase",
            )
            self.assertTrue(
                torch.equal(
                    sample.unpaired_e, torch.tensor(0, dtype=self.dtype_uhf)
                )  # incl. last batch
                or torch.equal(
                    sample.unpaired_e, torch.tensor([0, 0], dtype=self.dtype_uhf)
                ),
                msg="Inconsistent uhf for molecules generated from ase",
            )

    # FIXME: Outdated datareader structure
    def stest_dataloader_from_gfn2_fitset(self) -> None:
        """Test loading gfn2 fitset from disk and converting to geometry."""

        if not self.has_gfn2:
            return

        # read data from files
        data, file_list = Datareader.fetch_data(self.path)
        self.assertTrue(
            len(data) == 7217,  # 7217 + 3 (empty coord) + 87 (no-coord) = 7307
            msg="Not all data loaded from gfn2 fitset",
        )

        self.assertTrue(
            len(file_list) == 7217,  # 7217 + 3 (empty coord) + 87 (no-coord) = 7307
            msg="Incomplete file_list from gfn2 fitset",
        )

        # convert to geometry object
        geometry = Datareader.setup_geometry(data)
        self.assertListEqual(
            list(geometry.positions.shape),
            [7217, 240, 3],  # shape: (n_samples, n_atoms, 3)
            msg="Wrong padding for gfn2 fitset.",
        )
