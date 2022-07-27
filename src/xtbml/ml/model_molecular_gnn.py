import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Batch, LightningDataset
from typing import List, Dict
from torch import optim, nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as pygDataloader
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from .loss import WTMAD2Loss
from ..data.graph_dataset import MolecularGraph_Dataset
from .transforms import Pad_Hamiltonian
from ..typing import Tensor


class MolecularGNN(pl.LightningModule):
    """Graph based model for prediction on single molecules."""

    def __init__(self, cfg=None):
        super().__init__()

        # TODO: wrap into config dictionary
        nf = 5
        out = 1
        hidden = cfg.hidden if cfg.hidden else 10

        self.gconv1 = GCNConv(nf, hidden)
        self.gconv2 = GCNConv(hidden, out)

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

        # wandb logging
        # self.save_hyperparameters() # FIXME: issue when wandb.cfg is parsed

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

        return energies

    def training_step(self, batch: Batch, batch_idx: int):

        # prediction based on QM features
        y = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        loss = self.calc_loss(batch, y, y_true)

        # bookkeeping
        egfn1_mol = batch.egfn1.scatter_reduce(0, batch.batch, reduce="sum")
        egfn1_mol = torch.unsqueeze(egfn1_mol, -1)
        gfn1_loss = self.loss_fn(egfn1_mol, y_true)

        self.log("train_loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        self.log(
            "gfn1_loss", gfn1_loss.item(), on_epoch=True, batch_size=batch.num_graphs
        )

        # store y vs y_true
        if self.global_step % 10 == 0 and isinstance(self.logger, WandbLogger):
            data = [x for x in torch.cat((y, y_true), 1)]
            table = wandb.Table(data=data, columns=["y", "y_true"])
            self.logger.experiment.log(
                {"my_custom_id": wandb.plot.scatter(table, "y", "y_true")},
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

    # TODO: refactor into get config (maybe setup an individual config class?)
    #   * load config
    #   * get_logger() (define logger)
    #   * get_trainer()

    # TODO: add get_val_dataloader() based on split_train_test(range(dataset.len))

    # TODO: overfit single sample
    #       - does it work? -> nope (or only very unsufficient)

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--no_wandb", dest="no_wandb", default=False, action="store_true"
    )
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # print("args: ", args)

    # weights and biases interface
    if not args.no_wandb:
        args.logger = WandbLogger(project="dxtb", entity="hoelzerc")
    del args.no_wandb

    # some default settings
    args.max_epochs = 3
    args.log_every_n_steps = 1
    args.callbacks = [LearningRateMonitor(logging_interval="epoch")]

    return pl.Trainer.from_argparse_args(args)


def train_gnn():
    print("Train GNN")

    # model and training setup
    trainer = get_trainer()

    # for sweeps aka. hyperparameter searchs
    config_defaults = {"hidden": 50, "lr": 0.1, "channels": 16}
    wandb.config.update(config_defaults)

    model = MolecularGNN(wandb.config)

    # log gradients, parameter histogram and model topology
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.watch(model, log_freq=5, log="all")

    # data
    transforms = torch.nn.Sequential(
        Pad_Hamiltonian(n_shells=10),
    )

    dataset = MolecularGraph_Dataset(
        root=Path(__file__).resolve().parents[3] / "data" / "PTB",  # "GMTKN55"
        pre_transform=transforms,
    )

    # during dev only operate on subset
    dataset = torch.utils.data.Subset(dataset, range(10))
    print(len(dataset))
    print(type(dataset))

    train_loader = pygDataloader(
        dataset, batch_size=5, drop_last=False, shuffle=False, num_workers=8
    )

    # train model
    trainer.fit(model=model, train_dataloaders=train_loader)

    # To save pytorch lightning models with wandb, we use:
    # trainer.save_checkpoint('EarlyStoppingADam-32-0.001.pth')
    # wandb.save('EarlyStoppingADam-32-0.001.pth')
    # This creates a checkpoint file in the local runtime, and uploads it to wandb. Now, when we decide to resume training even on a different system, we can simply load the checkpoint file from wandb and load it into our program like so:
    # wandb.restore('EarlyStoppingADam-32-0.001.pth')
    # model.load_from_checkpoint('EarlyStoppingADam-32-0.001.pth')
    # Now the checkpoint has been loaded into the model and the training can be resumed using the desired training module.


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


def sweep():
    """Conduct a wandb hyperparameter search (sweep)."""

    # sweep aka hyperparameter search
    import wandb

    sweep_config = {
        "method": "grid",
        "parameters": {"hidden": {"values": [50, 100]}},
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_gnn)
