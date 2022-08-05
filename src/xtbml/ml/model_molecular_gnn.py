import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Batch, LightningDataset
from typing import List, Dict, Tuple
from torch import optim, nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as pygDataloader
from argparse import ArgumentParser
from pathlib import Path
import time

from pytorch_lightning.loggers import WandbLogger
import wandb

from xtbml.param import charge

from .loss import WTMAD2Loss
from ..data.graph_dataset import MolecularGraph_Dataset
from .transforms import Pad_Hamiltonian
from ..typing import Tensor
from .config import Lightning_Configuration
from ..param import GFN1_XTB as par
from ..xtb.calculator import Calculator
from ..wavefunction import mulliken
from ..basis.indexhelper import IndexHelper
from ..param.gfn1 import GFN1_XTB as par
from ..param.util import get_element_angular

import sys


class MolecularGNN(pl.LightningModule):
    """Graph based model for prediction on single molecules."""

    def __init__(self, cfg: dict):
        super().__init__()

        # TODO: wrap into config dictionary
        nf = 6  # 8
        out = 1
        hidden = cfg["hidden"]

        self.gconv1 = GCNConv(nf, hidden)
        self.gconv2 = GCNConv(hidden, out)
        self.gconvtmp = GCNConv(nf, out)

        # embedding
        self.node_embedding = None
        self.node_decoding = None  # torch.nn.Linear(out, 1) --> atomic energies
        self.node_embedding = torch.nn.Linear(nf, out)  # dummy for testing gradients
        self.node_decoding = torch.nn.Linear(out, out)  # dummy for testing gradients

        # misc
        self.activation_fn = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p=0.2)

        # loss
        self.loss_fn = nn.MSELoss(reduction="mean")

        # wandb logging
        self.save_hyperparameters()  # FIXME: issue when wandb.cfg is parsed

    def forward(self, batch: Batch):
        verbose = False

        if verbose:
            print("inside forward")
            print("before", batch.x.shape)

        ###############################
        # TODO: single sample (bs=1)

        # setup for backpropagation
        batch.pos.requires_grad_(True)
        print(batch.pos)

        ###
        def dummy_scf(batch):
            b = torch.tensor([[-2.1763, -0.4713, 212], [-0.6986, 1.3702, 43]])
            cdist = torch.cdist(
                batch.pos, b, p=2, compute_mode="use_mm_for_euclid_dist"
            )
            return {"energy": torch.sum(cdist, dim=1)}  # check cdist

        ###

        # calc singlepoint
        results = self.calc_xtb(batch)
        # results = dummy_scf(batch)

        # get SCF features (energies, population and charges)
        energies = results["energy"]
        """ihelp = IndexHelper.from_numbers(
            batch.numbers, get_element_angular(par.element)
        )
        pop = mulliken.get_atomic_populations(
            results["overlap"], results["density"], ihelp
        )
        charges = ihelp.reduce_orbital_to_atom(results["charges"])"""

        print("batch.x size", batch.x.shape)
        # add SCF features to batch
        batch.x = torch.cat(
            [
                batch.x,
                energies.unsqueeze(1),
                # pop.unsqueeze(1),
                # charges.unsqueeze(1),
            ],
            dim=1,
        )

        # normalise energies
        """energy_indices = torch.LongTensor(
            [
                0,
                1,
                2,
                5,
            ]
        )  # FIXME: dodgy indexing by order in node-features

        batch.x[:, energy_indices] = -1 * batch.x[:, energy_indices]
        batch.x[:, 2] = -1 * batch.x[:, 2]
        batch.x[:, energy_indices] = (
            batch.x[:, energy_indices]
            / batch.x[:, energy_indices].max(0, keepdim=True)[0]
        )"""
        # TODO: loosing information about total values -- better normalise over total dataset?

        """# calculate SCF
        data_list = batch.to_data_list()
        for sample in data_list:
            if verbose:
                print(sample.uid)

            # setup for backpropagation
            sample.pos.requires_grad_(True)

            # calc singlepoint
            results = self.calc_xtb(sample)

            # get SCF features (energies, population and charges)
            energies = results["energy"]
            ihelp = IndexHelper.from_numbers(
                sample.numbers, get_element_angular(par.element)
            )
            pop = mulliken.get_atomic_populations(
                results["overlap"], results["density"], ihelp
            )
            charges = ihelp.reduce_orbital_to_atom(results["charges"])

            # add SCF features to batch
            sample.x = torch.cat(
                [
                    sample.x,
                    energies.unsqueeze(1),
                    pop.unsqueeze(1),
                    charges.unsqueeze(1),
                ],
                dim=1,
            )

            # normalise energies
            energy_indices = torch.LongTensor(
                [
                    0,
                    1,
                    2,
                    5,
                ]
            )  # FIXME: dodgy indexing by order in node-features

            sample.x[:, energy_indices] = -1 * sample.x[:, energy_indices]
            sample.x[:, 2] = -1 * sample.x[:, 2]
            sample.x[:, energy_indices] = (
                sample.x[:, energy_indices]
                / sample.x[:, energy_indices].max(0, keepdim=True)[0]
            )
            # TODO: loosing information about total values -- better normalise over total dataset?

            if verbose:
                print(sample.x)

            if verbose and False:
                # print("results: ", results)
                # print("energy: ", energy)
                # print("gradient: ", gradient)
                print(
                    "results: ",
                    results["charges"].shape,
                    results["energy"].shape,
                    results["density"].shape,
                    results["hcore"].shape,
                    results["hamiltonian"].shape,
                    results["overlap"].shape,
                )

        batch = batch.from_data_list(data_list)"""
        # NOTE: this breaks the graph, i.e. gradients only
        #       available within data_list, but not in batch

        # embedding
        # TODO: add embedding layer for node features

        # TODO: add decoding layer for node features

        """x = self.node_embedding(batch.x)"""  # dummy for testing gradients

        """x = self.gconv1(x, batch.edge_index)
        # x = self.gconv1(batch.x, batch.edge_index)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.gconv2(x, batch.edge_index)"""

        """if verbose:
            print("after", x.shape)

        # sum over atomic energies to get molecular energies
        energies = global_add_pool(x=x, batch=batch.batch)  # [bs, 1]"""

        """energies = self.node_decoding(energies)  # dummy for testing gradients"""

        ##########

        energies = self.node_embedding(batch.x[0].unsqueeze(0))

        start_force = time.time()

        # calculate force based on autograd AD
        force = torch.autograd.grad(
            energies,
            batch.pos,
            grad_outputs=torch.ones_like(energies),
            create_graph=True,
        )
        # TODO:ensure to set opt.zero_grad() before

        assert len(force) == 1
        print("time needed for Force calculation: ", time.time() - start_force)

        gradients = force[0]

        return energies, gradients

    def backward(self, loss, optimizer, optimizer_idx):
        """Custom backward implementation."""
        verbose = False

        # NOTE: important to get **all leaf tensors** of the NN,
        #       otherwise gradient propagation is incorrect
        nn_leaf_tensors = [t for t in self.node_embedding.parameters()]

        if verbose:
            print("BEFORE: nn_leaf_tensors.grad", nn_leaf_tensors[0].grad)
            print("BEFORE: nn_leaf_tensors.grad", nn_leaf_tensors[1].grad)

        # only propagating gradient through model parameters (until beginning of ML model)
        loss.backward(inputs=nn_leaf_tensors)
        # ensure that loss is not backpropagated
        # until positions but only until NN parameters

        if verbose:
            print("AFTER: nn_leaf_tensors.grad", nn_leaf_tensors[0].grad)
            print("AFTER: nn_leaf_tensors.grad", nn_leaf_tensors[1].grad)

        # reminder that positions should not receive a gradient
        # assert sample.pos.grad == None
        assert nn_leaf_tensors[0].grad != None  # weights
        assert nn_leaf_tensors[1].grad == None  # bias

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        verbose = False

        # 1. train on energies --> E_ML
        # 2. calculate loss based on gradient (g_ml - g_ref)

        # prediction based on QM features
        y, gradients = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        # calculate loss
        gref = batch.gref
        force = gradients
        # use gradient for loss calculation
        loss = self.loss_fn(force, gref)
        if verbose:
            print(force.shape, gref.shape)
            print("Force", force)
            print("Reference", gref)
            # propagate gradients for updating model parameters
            print("loss", loss)
            print("loss.has_grad", loss.requires_grad)

        # TODO: incorporate into testsuite
        torch.autograd.gradcheck(
            self.loss_fn,
            (force.double(), gref.double()),
            check_backward_ad=True,
        )

        # TODO: maybe easier to debug in batch_size = 1 scenario?

        print("checking loss sizes")
        # sys.exit(0)
        return loss
        ###

        for i, force in enumerate(gradients):
            gref = batch.get_example(i).gref
            print(force.shape, gref.shape)
            print("Force", force)
            print("Reference", gref)

            # calculate loss
            loss = self.loss_fn(force, gref)

            # propagate gradients for updating model parameters
            print("loss", loss)
            print("loss.has_grad", loss.requires_grad)

            # TODO: incorporate into testsuite
            torch.autograd.gradcheck(
                self.loss_fn,
                (force.double(), gref.double()),
                check_backward_ad=True,
            )

            print("weight update")
            print("node_embedding")
            print(self.node_embedding.weight)
            print(self.node_embedding.weight.grad)  # TODO: why is that zero?
            print(self.node_embedding.bias)
            print(self.node_embedding.bias.grad)
            print("node_decoding")
            print(self.node_decoding.weight)
            print(self.node_decoding.weight.grad)  # TODO: why is that zero?
            print(self.node_decoding.bias)
            print(self.node_decoding.bias.grad)

            # TODO: maybe easier to debug in batch_size = 1 scenario?

            print("something breaks the gradient (weight.grad should not be zero)")
            sys.exit(0)

            assert self.node_embedding.weight.grad != None
            assert self.node_decoding.weight.grad != None
            # NOTE: does not change, regardless whether sample.requires_grad(True/False)
            # TODO: check why this is the case

            assert self.node_embedding.bias.grad == None
            assert self.node_decoding.bias.grad == None
            # NOTE: since loss depends only on force and force only on non-constant parts of the NN
            #       resulting in not updating weights during optimisation (would require bias.grad != None)

        sys.exit(0)
        # also see: https://stackoverflow.com/questions/71294401/pytorch-loss-function-that-depends-on-gradient-of-network-with-respect-to-input

        #################################
        #################################

        # loss = self.calc_loss(batch, y, y_true)

        # bookkeeping
        egfn1_mol = batch.egfn1.scatter_reduce(0, batch.batch, reduce="sum")
        egfn1_mol = torch.unsqueeze(egfn1_mol, -1)
        gfn1_loss = self.loss_fn(egfn1_mol, y_true)

        self.log("train_loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        self.log(
            "gfn1_loss",
            gfn1_loss.item(),
            on_epoch=True,
            batch_size=batch.num_graphs,
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

    def calc_xtb(self, sample) -> Dict[str, Tensor]:

        start_scf = time.time()

        numbers = sample.numbers
        positions = sample.pos.clone()
        charges = sample.charges.float()

        # calculate xtb via SCF
        calc = Calculator(numbers, positions, par)
        results = calc.singlepoint(numbers, positions, charges, verbosity=0)

        print("time needed for SCF: ", time.time() - start_scf)

        return results


def get_trainer(cfg: dict) -> pl.Trainer:
    """Construct pytorch lightning trainer from argparse. Otherwise default settings.

    Args:
        cfg (dict): Configuration for obtaining trainer

    Returns:
        pl.Trainer: Pytorch lightning trainer instance
    """

    # update defaults
    defaults = pl.Trainer.default_attributes()
    defaults.update((k, cfg[k]) for k in defaults.keys() & cfg.keys())

    return pl.Trainer(**defaults)


def train_gnn():
    print("Train GNN")

    # TODO: add get_val_dataloader() based on split_train_test(range(dataset.len))
    #   * add config cli parsing -- see: https://github.com/mpkocher/pydantic-cli

    # TODO: overfit single sample
    #       - does it work? -> nope (or only very unsufficient)

    # configuration setup
    cfg = Lightning_Configuration()

    # weights and biases interface
    # for sweeps aka. hyperparameter searchs
    if cfg.train_no_wandb == False:
        cfg.train_logger = WandbLogger(project="dxtb", entity="hoelzerc")
        wandb.config.update(cfg.dict())

    # model and training setup
    trainer = get_trainer(cfg.get_train_cfg())
    model = MolecularGNN(cfg.get_model_cfg())

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
    dataset = torch.utils.data.Subset(dataset, range(3))
    print(len(dataset))
    print(type(dataset))

    train_loader = pygDataloader(
        dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=8
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
        Pad_Hamiltonian(n_shells=10),
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
