from copy import deepcopy
from pathlib import Path
from typing import Generator, Tuple
import pytest
import torch
import tempfile

from xtbml.data.dataset import ReactionDataset, SampleDataset, store_subsets_on_disk
from xtbml.data.reactions import Reaction, Reactions
from xtbml.data.samples import Sample, Samples


@pytest.fixture(scope="class", name="data")
def fixture_data() -> Generator[
    Tuple[Samples, Reactions, SampleDataset, ReactionDataset], None, None
]:
    print(
        "Loading JSON files for 'Samples', 'Reactions', 'SampleDataset' and 'ReactionDataset'..."
    )

    path_samples1 = Path(Path.cwd(), "data/GMTKN55/samples_ACONF.json")
    path_samples2 = Path(Path.cwd(), "data/PTB/samples_HE.json")
    path_reactions1 = Path(Path.cwd(), "data/GMTKN55/reactions_ACONF.json")

    samples = Samples.from_json(path_samples1)
    reactions = Reactions.from_json(path_reactions1)
    reaction_dataset = ReactionDataset.from_json(path_samples1, path_reactions1)
    sample_dataset = SampleDataset.from_json(path_samples2)

    yield samples, reactions, sample_dataset, reaction_dataset


class TestDataset:
    """Testing handling of batched Geometry objects."""

    @classmethod
    def setup_class(cls):
        print("")
        print(cls.__name__)

    def test_load(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test loading the JSON files containing the samples and reactions."""
        samples, reactions, sample_dataset, reaction_dataset = data

        assert isinstance(samples, Samples)
        assert isinstance(reactions, Reactions)
        assert isinstance(reaction_dataset, ReactionDataset)
        assert isinstance(sample_dataset, SampleDataset)

    def test_indexing(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test dunder methods for indexing/slicing and length."""
        samples, reactions, sample_dataset, reaction_dataset = data

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

        # test reaction dataset
        assert isinstance(reaction_dataset[:3], ReactionDataset)
        assert len(reaction_dataset[:2]) == 2

        # test sample dataset
        assert isinstance(sample_dataset[:3], SampleDataset)
        assert len(sample_dataset[:2]) == 2

    def test_dict(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test conversion to dictionary."""
        samples, reactions, _, _ = data

        # test samples
        d = samples[0].to_dict()
        assert "uid" in d.keys()

        # test reactions
        d = reactions[0].to_dict()
        assert "uid" in d.keys()

    def test_change_dtype(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test for setting `torch.dtype` for tensor class attributes."""
        samples, reactions, _, _ = data

        # test samples
        dtype = torch.float64
        samples = samples.type(dtype)
        sample: Sample = samples[0]

        assert sample.positions.dtype == dtype
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
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test for setting `torch.device` for tensor class attributes."""
        samples, reactions, _, _ = data

        # test samples
        device = "cpu"
        samples = samples.to(torch.device(device))
        sample: Sample = samples[0]

        assert sample.positions.device == torch.device(device)
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

    def test_save_subset(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        _, _, _, reaction_dataset = data

        aconf = reaction_dataset
        aconf.sort()

        # write to temporary directory
        with tempfile.TemporaryDirectory() as td:
            store_subsets_on_disk(reaction_dataset.copy(), Path(td), ["ACONF"])
            dataset2 = ReactionDataset.from_json(
                path_samples=Path(td, "samples.json"),
                path_reactions=Path(td, "reactions.json"),
            )
            dataset2.sort()

            assert aconf == dataset2

        # loading full dataset (all samples and reactions)
        # g21ip = dataset[575:611]
        # g21ip.sort()

        # with tempfile.TemporaryDirectory() as td:
        #     store_subsets_on_disk(dataset.copy(), Path(td), ["G21IP"])
        #     dataset3 = ReactionDataset.from_json(
        #         path_samples=Path(td, "samples.json"),
        #         path_reactions=Path(td, "reactions.json"),
        #     )
        #     dataset3.sort()

        #     assert g21ip.equal(dataset3)

    # def test_to_df(self, data: Tuple[Samples, Reactions, ReactionDataset]) -> None:
    #     _, _, dataset = data

    #     aconf = dataset  # [:15]
    #     aconf.sort()

    #     print("now pad")
    #     aconf.pad()
    #     print("done padding")

    #     df = aconf.to_df()
    #     print(df)

    def test_to_json(
        self, data: Tuple[Samples, Reactions, SampleDataset, ReactionDataset]
    ) -> None:
        """Test for saving the dataset to disk. Check for identical saving-loading."""
        _, _, sample_dataset, reaction_dataset = data

        # write to temporary directory
        with tempfile.TemporaryDirectory() as td:
            reaction_dataset.to_json(Path(td))
            dataset2 = ReactionDataset.from_json(
                path_samples=Path(td, "samples.json"),
                path_reactions=Path(td, "reactions.json"),
            )
            assert reaction_dataset == dataset2

        with tempfile.TemporaryDirectory() as td_sample:
            sample_dataset.to_json(Path(td_sample))
            dataset3 = SampleDataset.from_json(
                path_samples=Path(td_sample, "samples.json"),
            )
            assert sample_dataset == dataset3