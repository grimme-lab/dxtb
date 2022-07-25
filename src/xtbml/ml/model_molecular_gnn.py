import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_add_pool
from torch_geometric.data import Data, Batch, LightningDataset
from typing import List, Dict
from torch import optim, nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as pygDataloader
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.csv_logs import CSVLogger

from .loss import WTMAD2Loss
from ..data.graph_dataset import MolecularGraph_Dataset
from .transforms import Pad_Hamiltonian
from ..typing import Tensor


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class MolecularGNN(pl.LightningModule):
    """Graph based model for prediction on single molecules."""

    def __init__(self):
        super().__init__()

        # TODO: wrap into config dictionary
        nf = 5
        out = 1

        self.gconv1 = GCNConv(nf, 50)
        self.gconv2 = GCNConv(50, out)

        # embedding
        self.node_embedding = None
        self.node_decoding = None  # torch.nn.Linear(out, 1) --> atomic energies

        # misc
        self.activation_fn = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p=0.2)

        # loss
        wtmad2_path = Path(__file__).resolve().parents[3] / "data" / "GMTKN55"
        self.loss_fn = WTMAD2Loss(wtmad2_path, reduction="mean")
        # self.loss_fn = nn.L1Loss(reduction="mean")
        self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, batch: Batch):
        verbose = False

        if verbose:
            print("inside forward")
            print("before", batch.x.shape)

        # embedding
        # TODO: add embedding layer for node features

        # TODO: add decoding layer for node features

        x = self.gconv1(batch.x, batch.edge_index)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.gconv2(x, batch.edge_index)

        if verbose:
            print("after", x.shape)

        # sum over atomic energies to get molecular energies
        energies = global_add_pool(x=x, batch=batch.batch)  # [bs, 1]

        print("here we are")
        import sys

        sys.exit(0)

        return energies

    def training_step(self, batch: Batch, batch_idx: int):

        # prediction based on QM features
        y = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        print(y, y_true)

        loss = self.calc_loss(batch, y, y_true)

        # bookkeeping
        if isinstance(self.loss_fn, WTMAD2Loss):
            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batch.uid]
            # number of partners per reaction
            n_partner = torch.tensor([1] * batch.num_graphs)

            gfn1_loss = self.loss_fn(
                batch.egfn1.unsqueeze(-1), y_true, subsets, n_partner
            )
            gfn2_loss = self.loss_fn(
                batch.egfn2.unsqueeze(-1), y_true, subsets, n_partner
            )
        else:
            egfn1_mol = batch.egfn1.scatter_reduce(0, batch.batch, reduce="sum")
            egfn1_mol = torch.unsqueeze(egfn1_mol, -1)
            gfn1_loss = self.loss_fn(egfn1_mol, y_true)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        self.log("gfn1_loss", gfn1_loss.item(), batch_size=batch.num_graphs)

        self.log(
            "lr",
            get_lr(self.optimizers()),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Batch):

        y = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        loss = self.calc_loss(batch, y, y_true)

        self.log("val_loss", loss, on_epoch=True, batch_size=batch.num_graphs)

    @torch.no_grad()
    def test_step(self, batch: Batch, _):

        y = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        loss = self.calc_loss(batch, y, y_true)

        self.log("test_loss", loss, on_epoch=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        # TODO: set learning rate and optimiser (from cfg etc.)
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)

        scheduler_options = {
            "Const": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=1.0
            ),
            "StepLR": torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            ),
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, verbose=True
            ),
            "CyclicLR": torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=0.001,
                max_lr=0.1,
                step_size_up=2000,
                step_size_down=None,
                mode="triangular",
                gamma=1.0,
                cycle_momentum=False,
            ),
        }

        lr_schedulers = {
            "scheduler": scheduler_options["StepLR"],
            "monitor": "train_loss",
        }

        return [optimizer], [lr_schedulers]

    def calc_loss(self, batch: Batch, y: Tensor, y_true: Tensor) -> Tensor:
        # optimize model parameter and feature parameter
        if isinstance(self.loss_fn, WTMAD2Loss):
            # derive subset from partner list
            subsets = [s.split("/")[0] for s in batch.uid]
            # number of partners per reaction
            n_partner = torch.tensor([1] * batch.num_graphs)

            loss = self.loss_fn(y, y_true, subsets, n_partner)
        else:
            loss = self.loss_fn(y, y_true)

        return loss


def get_trainer() -> pl.Trainer:
    """Construct pytorch lightning trainer from argparse. Otherwise default settings.

    Returns:
        pl.Trainer: Pytorch lightning trainer instance
    """

    # parse arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # print("args: ", args)

    # some (temporary) default settings
    # args.profiler = "simple"
    args.max_epochs = 3
    args.log_every_n_steps = 5
    args.callbacks = [LearningRateMonitor(logging_interval="epoch")]
    args.logger = CSVLogger("logs", name="gnn", version=f"{args.max_epochs}")

    return pl.Trainer.from_argparse_args(args)


def train_gnn():
    print("Train GNN")

    # model and training setup
    trainer = get_trainer()
    model = MolecularGNN()

    # data
    transforms = torch.nn.Sequential(
        Pad_Hamiltonian(n_shells=300),
    )

    dataset = MolecularGraph_Dataset(
        root=Path(__file__).resolve().parents[3] / "data" / "PTB",  # "GMTKN55"
        pre_transform=transforms,
    )
    print(len(dataset))
    print(type(dataset))

    train_loader = pygDataloader(
        dataset, batch_size=10, drop_last=False, shuffle=False, num_workers=8
    )

    # train model
    trainer.fit(model=model, train_dataloaders=train_loader)


def test_gnn():
    print("Test GNN")

    # model and training setup
    trainer = get_trainer()
    model = MolecularGNN()

    # data
    transforms = torch.nn.Sequential(
        Pad_Hamiltonian(n_shells=300),
    )
    dataset = MolecularGraph_Dataset(
        root=Path(__file__).resolve().parents[3] / "data" / "PTB",  # "GMTKN55"
        pre_transform=transforms,
    )

    test_loader = pygDataloader(
        dataset, batch_size=10, drop_last=False, shuffle=False, num_workers=8
    )

    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path="./lightning_logs/version_109/checkpoints/epoch=0-step=1260.ckpt",
    )
    # ckpt_path="best" # load the best checkpoint automatically (lightning tracks this for you)

    # TODO: checkpoint_callback.best_model_path
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
    # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html
