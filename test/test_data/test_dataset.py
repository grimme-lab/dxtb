from pathlib import Path

import torch

from xtbml.data.dataset import Sample, Samples, Reaction, Reactions, ReactionDataset


class TestDataset:
    """Testing handling of batched Geometry objects."""

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    def test_load(self) -> None:
        """Test loading the JSON files containing the samples and reactions."""
        path_samples = Path(Path.cwd(), "data/features.json")
        path_reactions = Path(Path.cwd(), "data/reactions.json")

        samples = Samples.from_json(path_samples)
        assert type(samples) == Samples

        reactions = Reactions.from_json(path_reactions)
        assert type(reactions) == Reactions

        dataset = ReactionDataset.create_from_disk(path_reactions, path_samples)
        assert type(dataset) == ReactionDataset

    def test_indexing(self) -> None:
        """Test dunder methods for indexing/slicing and length."""
        path_samples = Path(Path.cwd(), "data/features.json")
        path_reactions = Path(Path.cwd(), "data/reactions.json")

        samples = Samples.from_json(path_samples)
        reactions = Reactions.from_json(path_reactions)

        # test samples
        sample = samples[0]
        assert type(sample) == Sample
        assert hasattr(sample, "uid") == True

        assert type(samples[:3]) == list
        assert len(samples[:2]) == 2

        # test reactions
        reaction = reactions[0]
        assert type(reaction) == Reaction
        assert hasattr(reaction, "uid") == True

        assert type(reactions[:3]) == list
        assert len(reactions[:2]) == 2

    def test_dict(self) -> None:
        path_samples = Path(Path.cwd(), "data/features.json")
        path_reactions = Path(Path.cwd(), "data/reactions.json")

        samples = Samples.from_json(path_samples)
        reactions = Reactions.from_json(path_reactions)

        # test samples
        d = samples[0].to_dict()
        assert "uid" in d.keys()

        # test reactions
        d = reactions[0].to_dict()
        assert "uid" in d.keys()

    def test_change_dtype(self) -> None:
        """Test for setting `torch.dtype` for tensor class attributes."""
        path_samples = Path(Path.cwd(), "data/features.json")
        path_reactions = Path(Path.cwd(), "data/reactions.json")

        samples = Samples.from_json(path_samples)
        reactions = Reactions.from_json(path_reactions)

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

    def test_change_device(self) -> None:
        """Test for setting `torch.device` for tensor class attributes."""
        path_samples = Path(Path.cwd(), "data/features.json")
        path_reactions = Path(Path.cwd(), "data/reactions.json")

        samples = Samples.from_json(path_samples)
        reactions = Reactions.from_json(path_reactions)

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

    
    def test_singlepoint(self) -> None:
        """Test for (slow) on-the-fly feature generation."""
        path_samples = Path(Path.cwd(), "data/features.json")

        samples: Samples = Samples.from_json(path_samples)
        sample: Sample = samples[0]
        
        h0, ovlp, cn = sample.calc_singlepoint()
        
        print(h0, ovlp, cn)
        
        