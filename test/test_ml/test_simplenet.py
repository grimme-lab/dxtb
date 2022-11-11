from pathlib import Path
import torch

from dxtb.data.dataset import get_gmtkn55_dataset
from dxtb.ml.loss import WTMAD2Loss
from dxtb.ml.model import Simple_Net




class TestWTMAD2Loss:
    """Testing the loss calulation based on GMTKN55 weighting."""

    path = Path(Path(__file__).resolve().parents[2], "data", "GMTKN55")
    """Absolute path to fit set data."""

    def setup_class(self):
        self.dataset = get_gmtkn55_dataset(
            self.path, file_reactions="reactions.json", file_samples="samples.json"
        )

        self.loss_fn = WTMAD2Loss(self.path)

    def basic_training(self, simplicity):
        """Baseline for conducting training."""

        model = Simple_Net(simplicity)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        dl = self.dataset.get_dataloader({"batch_size": 1, "shuffle": False})

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
        return losses

    def test_simplicity0(self):
        """Model return Eref."""
        losses = self.basic_training(simplicity=0)
        assert losses == 0.0

