from pathlib import Path
from typing import Generator, Tuple
import pytest
import torch
import tempfile

from xtbml.data.dataset import ReactionDataset, store_subsets_on_disk
from xtbml.data.reactions import Reaction, Reactions
from xtbml.data.samples import Sample, Samples


@pytest.fixture(scope="class")
def data() -> Generator[Tuple[Samples, Reactions, ReactionDataset], None, None]:
    print("Loading JSON files for 'Samples', 'Reactions' and 'ReactionDataset'...")

    path_samples = Path(Path.cwd(), "data/samples.json")
    path_reactions = Path(Path.cwd(), "data/reactions.json")

    samples = Samples.from_json(path_samples)
    reactions = Reactions.from_json(path_reactions)
    dataset = ReactionDataset.create_from_disk(path_reactions, path_samples)

    yield samples, reactions, dataset

    # print("Teardown 'Loading'.")


class TestDataset:
    """Testing handling of batched Geometry objects."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def test_load(self, data: Tuple[Samples, Reactions, ReactionDataset]) -> None:
        """Test loading the JSON files containing the samples and reactions."""
        samples, reactions, dataset = data

        assert isinstance(samples, Samples)
        assert isinstance(reactions, Reactions)
        assert isinstance(dataset, ReactionDataset)

    def test_indexing(self, data: Tuple[Samples, Reactions, ReactionDataset]) -> None:
        """Test dunder methods for indexing/slicing and length."""
        samples, reactions, dataset = data

        # test samples
        sample = samples[0]
        assert isinstance(sample, Sample)
        assert hasattr(sample, "uid") is True

        assert isinstance(samples[:3], list)
        assert len(samples[:2]) == 2

        # test reactions
        reaction = reactions[0]
        assert isinstance(reaction, Reaction)
        assert hasattr(reaction, "uid") is True

        assert isinstance(reactions[:3], list)
        assert len(reactions[:2]) == 2

        # test dataset
        assert isinstance(dataset[:3], ReactionDataset)
        assert len(dataset[:2]) == 2

    def test_dict(self, data: Tuple[Samples, Reactions, ReactionDataset]) -> None:
        """Test conversion to dictionary."""
        samples, reactions, _ = data

        # test samples
        d = samples[0].to_dict()
        assert "uid" in d.keys()

        # test reactions
        d = reactions[0].to_dict()
        assert "uid" in d.keys()

    def test_change_dtype(
        self, data: Tuple[Samples, Reactions, ReactionDataset]
    ) -> None:
        """Test for setting `torch.dtype` for tensor class attributes."""
        samples, reactions, _ = data

        # test samples
        dtype = torch.float64
        samples = samples.type(dtype)
        sample: Sample = samples[0]

        assert sample.xyz.dtype == dtype
        assert sample.numbers.dtype != dtype
        assert sample.egfn1.dtype == dtype
        assert sample.ovlp.dtype == dtype
        assert sample.h0.dtype == dtype
        assert sample.cn.dtype == dtype

        # test reactions
        dtype = torch.float64
        reactions = reactions.type(dtype)
        reaction: Reaction = reactions[0]

        assert reaction.nu.dtype != dtype
        assert reaction.egfn1.dtype == dtype
        assert reaction.eref.dtype == dtype

    def test_change_device(
        self, data: Tuple[Samples, Reactions, ReactionDataset]
    ) -> None:
        """Test for setting `torch.device` for tensor class attributes."""
        samples, reactions, _ = data

        # test samples
        device = "cpu"
        samples = samples.to(torch.device(device))
        sample: Sample = samples[0]

        assert sample.xyz.device == torch.device(device)
        assert sample.numbers.device == torch.device(device)
        assert sample.egfn1.device == torch.device(device)
        assert sample.ovlp.device == torch.device(device)
        assert sample.h0.device == torch.device(device)
        assert sample.cn.device == torch.device(device)

        # test reactions
        device = "cpu"
        reactions = reactions.to(torch.device(device))
        reaction: Reaction = reactions[0]

        assert reaction.nu.device == torch.device(device)
        assert reaction.egfn1.device == torch.device(device)
        assert reaction.eref.device == torch.device(device)

    def stest_singlepoint(
        self, data: Tuple[Samples, Reactions, ReactionDataset]
    ) -> None:
        """Test for (slow) on-the-fly feature generation."""
        samples, _, _ = data

        sample: Sample = samples[0]

        h0, ovlp, cn = sample.calc_singlepoint()

        print(h0, ovlp, cn)

    def test_save_subset(
        self, data: Tuple[Samples, Reactions, ReactionDataset]
    ) -> None:
        _, _, dataset = data

        aconf = dataset[:15]
        aconf.sort()

        g21ip = dataset[575:611]
        g21ip.sort()

        # write to temporary directory
        with tempfile.TemporaryDirectory() as td:
            store_subsets_on_disk(dataset.copy(), Path(td), ["ACONF"])
            dataset2 = ReactionDataset.create_from_disk(
                path_reactions=Path(td, "reactions.json"),
                path_samples=Path(td, "samples.json"),
            )
            dataset2.sort()

            assert aconf.equal(dataset2)

            store_subsets_on_disk(dataset.copy(), Path(td), ["G21IP"])
            dataset3 = ReactionDataset.create_from_disk(
                path_reactions=Path(td, "reactions.json"),
                path_samples=Path(td, "samples.json"),
            )
            dataset3.sort()

            assert g21ip.equal(dataset3)

    def stest_to_json(self, data: Tuple[Samples, Reactions, ReactionDataset]) -> None:
        """Test for saving the dataset to disk. Check for identical saving-loading."""
        __, __, dataset = data

        # write to temporary directory
        with tempfile.TemporaryDirectory() as td:
            dataset.to_disk(Path(td))
            dataset2 = ReactionDataset.create_from_disk(
                path_reactions=Path(td, "reactions.json"),
                path_samples=Path(td, "samples.json"),
            )
            assert dataset.equal(dataset2)
