from pathlib import Path
import pytest
import torch

from xtbml.ml.loss import WTMAD2Loss
from xtbml.data.dataset import ReactionDataset, get_gmtkn_dataset

from .gmtkn55 import GMTKN55


class TestWTMAD2Loss:
    """Testing the loss calulation based on GMTKN55 weighting."""

    path = "./data"

    def setup_class(self):
        print()
        print("Test custom loss function")
        self.dataset = get_gmtkn_dataset(self.path)

        # subsets in GMTKN55
        self.all_sets = set([r.uid.split("_")[0] for r in self.dataset.reactions])

        # loss function
        self.loss_fn = WTMAD2Loss(self.path)

    def teardown_class(self):
        # teardown_class called once for the class
        pass

    def setup_method(self):
        # setup_method called for every method
        pass

    def teardown_method(self):
        # teardown_method called for every method
        pass

    def test_data(self):

        # datatype
        assert isinstance(self.dataset, ReactionDataset)

        # number of reactions in GMTKN-55
        assert len(self.dataset) == 1505

        # number of subsets in GMTKN-55
        assert len(self.all_sets) == 55

    def test_naming_consistent(self):
        for r in self.dataset.reactions:
            subset = r.uid.split("_")[0]
            if subset == "BH76RC":
                subset = "BH76"
            partners = [s.split("/")[0] for s in r.partners]
            assert {subset} == set(partners), "Partner and reaction naming inconsistent"

    def test_loading(self):
        """Check consistency of GMTKN55 through by comparing with hard-coded averages and counts (number of reactions) of subsets."""
        atol = 1.0e-2
        rtol = 1.0e-4
        TOTAL_AVG = 57.82

        for (_, target), (_, ref) in zip(self.loss_fn.subsets.items(), GMTKN55.items()):
            # counts (type and value)
            assert ref["count"].is_integer()
            assert target["count"].item().is_integer()
            assert int(ref["count"]) == int(target["count"])

            # averages
            assert torch.allclose(
                torch.tensor(ref["avg"]),
                target["avg"],
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

        # total average
        assert torch.allclose(
            self.loss_fn.total_avg,
            torch.tensor(TOTAL_AVG),
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )

        # total len
        assert len(self.loss_fn.subsets) == 55

    def test_single(self):
        n_reactions = 2.0
        n_reactions_i = 2
        input = torch.arange(n_reactions, requires_grad=True)
        target = torch.arange(n_reactions) + 3
        # partner subsets
        label = ["ACONF"] * n_reactions_i * int(n_reactions)
        # reaction lengths
        n_partner = torch.tensor([n_reactions_i, n_reactions_i])

        self.loss_fn.reduction = "mean"
        output = self.loss_fn(input, target, label, n_partner)

        assert output.shape == torch.Size([])
        assert output.item() == 25.791534423828125

    @pytest.mark.grad
    def test_grad(self):
        # NOTE: currently no custom backward() functionality implemented

        n_reactions = 2.0
        n_reactions_i = 2
        # NOTE: requires grad=True and double precision
        input = torch.arange(n_reactions, requires_grad=True, dtype=torch.float64)
        target = torch.arange(n_reactions, requires_grad=True, dtype=torch.float64) + 3
        label = ["ACONF"] * n_reactions_i * int(n_reactions)
        n_partner = torch.tensor([n_reactions_i, n_reactions_i])

        assert torch.autograd.gradcheck(
            WTMAD2Loss(self.path),
            (input, target, label, n_partner),
            raise_exception=True,
        )

    @pytest.mark.parametrize("batchsize", [1, 2, 10])
    def test_gmtkn(self, batchsize):
        # TODO: compare subset and total WTMAD with paper and with implementation in evaluation.py

        print("\nload dl with bs", batchsize)
        dl = self.dataset.get_dataloader({"batch_size": batchsize, "shuffle": True})
        self.loss_fn.reduction = "sum"

        loss = torch.tensor([0.0])
        for i, (_, batched_reaction) in enumerate(dl):

            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            loss += self.loss_fn(
                batched_reaction.egfn1, batched_reaction.eref, subsets, n_partner
            )

        assert torch.allclose(
            loss,
            torch.tensor([44728.2891]),
            rtol=1.0e-4,
            atol=1.0e-6,
            equal_nan=False,
        )

    def test_gmtkn_subsets(self):

        dl = self.dataset.get_dataloader({"batch_size": 2, "shuffle": False})
        self.loss_fn.reduction = "none"

        losses = {k: torch.tensor([0.0]) for k in self.all_sets}

        # calc loss per subset
        for i, (_, batched_reaction) in enumerate(dl):
            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batched_reaction.partners]
            # different number of partners per reaction
            n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

            loss = self.loss_fn(
                batched_reaction.egfn1, batched_reaction.eref, subsets, n_partner
            )

            label = [subsets[i] for i in torch.cumsum(n_partner, dim=0) - 1]

            # add each loss to corresponding subset
            for j in range(len(label)):
                losses[label[j]] += loss[j]

        # normalise loss per subset (optional)
        for k, v in losses.items():
            losses[k] = v * len(self.dataset) / self.loss_fn.total_avg

        # print(losses)

        # TODO: compare subset and total WTMAD with paper and with implementation in evaluation.py
        # assert torch.allclose(
        #    loss,
        #    torch.tensor([44055.55078125]),
        #    rtol=1.0e-4,
        #    atol=1.0e-6,
        #    equal_nan=False,
        # )
