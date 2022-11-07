from pathlib import Path
from typing import List
import pandas as pd
import pytest
import torch
from dxtb.ml.evaluation import evaluate

from dxtb.data.dataset import get_gmtkn55_dataset
from dxtb.ml.loss import WTMAD2Loss
from dxtb.ml.model import Simple_Net


from .gmtkn55 import GMTKN55


class TestWTMAD2Loss:
    """Testing the loss calulation based on GMTKN55 weighting."""

    path = Path(Path(__file__).resolve().parents[2], "data")
    """Absolute path to fit set data."""

    def setup_class(self):
        print("Test Simple_Net model")
        self.dataset = get_gmtkn55_dataset(self.path)

        self.loss_fn = WTMAD2Loss(self.path)

    def basic_training(self, simplicity):
        """Baseline for conducting training."""

        model = Simple_Net(simplicity)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        dl = self.dataset.get_dataloader({"batch_size": 1, "shuffle": True})

        losses = []
        wtm2_ref = [], []
        losses_epoch = []
        mads_epoch, mads_ref_epoch, wtm2_ref_epoch = [], [], []
        for batched_samples, batched_reaction in dl:
            # predict
            y = model(batched_samples, batched_reaction)
            y_true = batched_reaction.eref

            # calc loss
            if isinstance(self.loss_fn, WTMAD2Loss):
                subsets = [s.split("/")[0] for s in batched_reaction.partners]
                n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

                loss = self.loss_fn(y, y_true, subsets, n_partner)
            else:
                loss = self.loss_fn(y, y_true)

            # optimize model parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # bookkeeping
            losses_epoch.append(loss.item())
            if isinstance(self.loss_fn, WTMAD2Loss):
                wtm2_ref_epoch.append(
                    self.loss_fn(
                        batched_reaction.egfn1, y_true, subsets, n_partner
                    ).item()
                )

        losses = sum(losses_epoch) / len(losses_epoch)
        wtm2_ref = sum(wtm2_ref_epoch) / len(wtm2_ref_epoch)
        print(f"Loss: {losses} WTM2_REF: {wtm2_ref}")
        return losses

    def test_simplicity0(self):
        """Model return Eref."""
        losses = self.basic_training(simplicity=0)
        assert losses == 0.0

    def test_simplicity1(self):
        """Model return Egfn1."""
        losses = self.basic_training(simplicity=1)
        gmtkn55 = 29.719778135001807
        assert losses == gmtkn55

    def test_simplicity2(self):
        """Model use sample Egfn1."""
        losses = self.basic_training(simplicity=2)

        # TODO: not implemented yet
        # assert losses == gmtkn55
