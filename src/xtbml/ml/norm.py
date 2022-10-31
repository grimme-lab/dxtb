from pathlib import Path

import torch

from xtbml.data.samples import Sample

from ..data.dataset import ReactionDataset
from ..typing import Any, Tensor

# NOTE: THIS IS NOT PRODUCTION READY


class Normalisation:
    def __init__(
        self,
        path: str | Path | None = None,  # TODO:
        skip_reaction_params: list[str] = ["eref", "egfn1"],
        skip_sample_params: list[str] = ["xyz", "egfn1"],
    ) -> None:
        """Module containing normalisation functionality.

        Parameters
        ----------
        path : Union[None, str, Path], optional
            Path for loading or saving normalisation, by default None
        skip_reaction_params : List[str], optional
            Reaction attributes to skip for calculation and application of normalisation, by default ["eref", "egfn1"]
        skip_sample_params : List[str], optional
            Sample attributes to skip for calculation and application of normalisation, by default ["xyz", "egfn1"]
        """

        # TODO: add functionality to read in normalisation factors from disk
        # TODO: add functionality to store normalisation factors to disk
        self.path = path

        self.skip_r = skip_reaction_params
        self.skip_s = skip_sample_params

    def normalise(
        self,
        dataset: ReactionDataset,
    ) -> tuple[dict[str, dict[str, Tensor]]]:
        # Normalise dataset in place.
        # use basic standard deviation
        # NOTE: optionally returns the normalisation factors

        # NOTE: normalisation is conducted over all reactions - and not samples!

        loader = dataset.get_dataloader({"batch_size": len(dataset), "num_workers": 1})
        data = next(iter(loader))

        # TODO
        dataset.prune()

        # batch all samples in reaction to one single sample
        for j, sample in enumerate(data[0]):
            print(j)
            if j == 0:
                batched_sample = sample.to_dict()
                batched_sample["uid"] = f"BATCH {j}"

                for k, v in batched_sample.items():
                    if isinstance(v, Tensor):
                        batched_sample[k] = v.unsqueeze(0)
                continue
            print(batched_sample["egfn1"])
            print(batched_sample["egfn1"].shape)

            for k, v in sample.to_dict().items():
                if not isinstance(v, Tensor):
                    continue
                # add to single sample
                batched_sample[k] = torch.concat(
                    (batched_sample[k].squeeze(), v), dim=0
                )

        s = Sample(**batched_sample)
        print(s.egfn1.shape)
        print(len(dataset.samples))
        print(len(dataset))
        # TODO: add test

        # NOTE: merge all _flattened_ features together
        for j, sample in enumerate(dataset.samples):
            # batch together
            if j == 0:
                batched_sample = sample.to_dict()
                batched_sample["uid"] = f"BATCH {j}"

                for k, v in batched_sample.items():
                    if isinstance(v, Tensor):
                        if len(v.shape) == 0:
                            batched_sample[k] = v.unsqueeze(0)
                        else:
                            batched_sample[k] = v.flatten()

                continue

            for k, v in sample.to_dict().items():
                if not isinstance(v, Tensor):
                    continue

                print("batching", k)

                if len(v.shape) == 0:
                    v = v.unsqueeze(0)
                v = v.flatten()
                print(v.shape)
                # add to single sample
                batched_sample[k] = torch.concat((batched_sample[k], v), dim=0)

        # print(batched_sample["egfn1"])
        print(batched_sample["egfn1"].shape)
        print(batched_sample["h0"].shape)
        print(dataset.samples)
        print(len(dataset.samples))

        # TODO: normalisation should be conducted sample wise
        #       esp. to conserve permutation invariance
        # assert s.egfn1.shape == len(dataset)

        # TODO: for correct normalisation, no padding should be applied

        # normalisation factors for reactions and samples
        r_norm = Normalisation.calc_norm(data[1], self.skip_r)
        s_norm = Normalisation.calc_norm(s, self.skip_s)
        print(r_norm)
        print(s_norm)

        # TODO: apply this to the original dataset
        """Normalisation.apply_norm(data[1], r_norm)

        print("SAMPLES")

        for i, s in enumerate(data[0]):
            print("BEFORE", i, s.egfn1)

        for i, s in enumerate(data[0]):
            apply_norm(s, s_norm, skip=["xyz"])

            # TODO: set sample back into dataset
            # better: update dynamically

        for i, s in enumerate(data[0]):
            print("After", i, s.egfn1)"""
        # TODO: maybe use in tests as well

        # apply normalisation
        print("\n CHECKING real dataset now")

        print("BEFORE")
        a, b = 1, 1  # TODO: for 4, 1 this does not look right for samples
        print(dataset[a][1].egfn1)
        print(dataset[a][0][b].egfn1)

        for i in range(len(dataset.reactions)):
            r = dataset.reactions[i]
            Normalisation.apply_norm(r, r_norm, self.skip_r)

        for i in range(len(dataset.samples)):
            s = dataset.samples[i]
            Normalisation.apply_norm(s, s_norm, self.skip_s)

        print("AFter")
        print(dataset[a][1].egfn1)
        print(dataset[a][0][b].egfn1)
        # TODO: add test (calculating by hand and with method)

        # print(s_norm["egfn1"])
        # TODO: is this correct?
        #   because the dataset is not iterable
        #   --> replace for i, s in enumerate(data[0]): to for i in enumerate(data[0]): dataset[i]

        # TODO: batching of dataloader should not affect the normalisation! --> write test for this
        # i.e. manually adding std and mean up and then compare for bs = all vs bs = 1
        # https://deeplizard.com/learn/video/lu7TCu7HeYc

        # TODO: make this a function for dataset class
        #       or add to a transforms class

        # TODO: as this takes a while, save to disk
        #       ideally have a functionality to write to disk

        print("r_norm")
        print(r_norm)
        print("s_norm")
        print(s_norm)

        print("include normalisation")
        return r_norm, s_norm

    # calc normalisation factors
    def calc_norm(
        obj, skip=[], dtypes=[torch.float32, torch.float64]
    ) -> dict[str, dict[str, Tensor]]:
        # calculate normalisation factor for tensor based object
        obj_norm = {}

        for slot, attr in Normalisation.yield_slot_attributes(obj, skip, dtypes):

            # calculate normalisation factor
            mean, std = attr.mean(), attr.std()
            # TODO: maybe add sklearn normalisers here

            # store normalisation
            obj_norm[slot] = {"mean": mean, "std": std}

        return obj_norm

    def apply_norm(obj, norm: dict, skip=[], dtypes=[torch.float32, torch.float64]):

        for slot, attr in Normalisation.yield_slot_attributes(
            obj, skip=skip, dtypes=dtypes
        ):
            # print(slot)
            # print(obj.__getattribute__(slot))
            # TODO: add sklearn here
            obj.__setattr__(slot, (attr - norm[slot]["mean"]) / norm[slot]["std"])
            # print(obj.__getattribute__(slot))

    def yield_slot_attributes(obj: Any, skip: list, dtypes: list) -> tuple[str, Tensor]:
        """Generator that yields all slotted attributes of obj

        Args:
            obj (Any): Object to be iterated over.
            skip (list): Slots to be skipped
            dtypes (list): Dtypes of the slots

        Yields:
            tuple: Tuple of slot name and slot value.
        """
        # TODO: move method to utils

        for slot in obj.__slots__:

            # skip internal attributes
            if slot.startswith("__") or slot in skip:
                continue

            attr = getattr(obj, slot)

            # skip non-numerical attributes
            if not isinstance(attr, Tensor):
                continue
            if not attr.dtype in dtypes:
                continue
            yield slot, attr
