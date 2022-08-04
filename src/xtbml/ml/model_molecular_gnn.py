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
        nf = 8
        out = 1
        hidden = cfg["hidden"]

        self.gconv1 = GCNConv(nf, hidden)
        self.gconv2 = GCNConv(hidden, out)

        # embedding
        self.node_embedding = None
        self.node_decoding = None  # torch.nn.Linear(out, 1) --> atomic energies

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
        # calculate SCF
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

        batch = batch.from_data_list(data_list)
        # NOTE: this breaks the graph, i.e. gradients only
        #       available within data_list, but not in batch

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

        ##########

        # calculate gradient
        gradients = []
        for i, sample in enumerate(data_list):
            if verbose:
                print(i, sample.uid)

            start_grad = time.time()
            """energies[i].backward(retain_graph=True)
            gradient = sample.pos.grad"""

            ##############
            print("time needed for GRAD: ", time.time() - start_grad)

            print(energies[i].shape)

            # calculate force based on autograd AD
            force = torch.autograd.grad(
                energies[i],
                # energies,
                sample.pos,
                grad_outputs=torch.ones_like(energies[i]),
                # grad_outputs=torch.ones_like(energies),
                # retain_graph=True,
                create_graph=True,
                # is_grads_batched=True,  # Not working, since sample.pos have different shapes
            )
            # NOTE: in real application set opt.zero_grad() before

            print("energies", energies)
            # print("gradient", gradient)
            print("force", force)  # this has a gradient!

            print("energies", energies.shape)
            print("energies[i]", energies[i].shape)
            # print("gradient", gradient.shape)
            print("force", force[0].shape)
            # TODO: probably need double() precision here!

            # TODO: check whether force changes if no prev cacl is done

            """energies tensor([[-9.0825],
                    [-3.5994],
                    [-7.0515]], grad_fn=<ScatterAddBackward0>)
            gradient tensor([[-0.0012,  0.0021,  0.0085],
                    [ 0.0004, -0.0008, -0.0035],
                    [-0.0004, -0.0011, -0.0025],
                    [ 0.0012, -0.0002, -0.0025]])
            force (tensor([[-0.0012,  0.0021,  0.0085],
                    [ 0.0004, -0.0008, -0.0035],
                    [-0.0004, -0.0011, -0.0025],
                    [ 0.0012, -0.0002, -0.0025]], grad_fn=<AddBackward0>),)"""

            if i != 1:
                continue
            sys.exit(0)
            ##############
            if verbose:
                print("time needed for GRAD: ", time.time() - start_grad)

                print("gradient", gradient)
                sys.exit(0)

                assert gradient != None

                # TODO: solve this for backpropagation and NN model update
                print(sample.pos.requires_grad)
                print(sample.pos.grad.requires_grad)
                print(gradient.requires_grad)
                # see: https://discuss.pytorch.org/t/error-by-recursively-calling-jacobian-in-a-for-loop/125924
                #       https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
                #       https://stackoverflow.com/questions/64997817/how-to-compute-hessian-of-the-loss-w-r-t-the-parameters-in-pytorch-using-autogr

            # required for backpropagation
            gradient.requires_grad_(True)
            # TODO: how to allow for correct backpropagation

            gradients.append(gradient)

        ##########

        return energies, gradients

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        # 1. train on energies --> E_ML
        # 2. calculate loss based on gradient (g_ml - g_ref)

        # prediction based on QM features
        y, gradients = self(batch)
        y_true = torch.unsqueeze(batch.eref, -1)

        # use gradient for loss calculation
        # TODO: how to calculate batchwise loss? -- looping as first option
        loss = torch.zeros([len(gradients)])
        for i, gradient in enumerate(gradients):
            # TODO: check gradient remains intact

            # print("grad functions: ", y.grad_fn, gradient.grad_fn)
            loss_e = 10e-15 * self.loss_fn(y, batch.get_example(i).eref)
            loss_g = self.loss_fn(gradient, batch.get_example(i).gref)
            loss[i] = loss_e + loss_g

            print("loss_e", loss_e)
            print("loss_g", loss_g)
            print("loss[i]", loss[i])
        # TODO: loss needs to have correct gradient
        #       -- gradient of gradient needs to update NN params

        # TODO: build a simple test case to verify that gradient calculation (for forces and loss) works!

        loss = torch.mean(loss)
        print("loss", loss)

        #################################
        #################################
        # ensure that loss is not backpropagated
        # until positions but only until NN parameters
        sample.a.requires_grad_(False)
        sample.b.requires_grad_(False)

        # propagate gradients for updating model parameters

        # 6. calculate loss and update model parameters
        loss = self.loss_fn(force[0], grad_ref)  # loss tensor(184856.5000)
        # also see: https://stackoverflow.com/questions/71294401/pytorch-loss-function-that-depends-on-gradient-of-network-with-respect-to-input

        print("loss", loss)  # loss tensor(184856.5000, tangent=53820.0)
        print("loss.has_grad", loss.requires_grad)

        assert model.layer.weight.grad != None  # tensor([[ 7812., 46008.]])
        # NOTE: does not change, regardless whether sample.requires_grad(True/False)
        # TODO: check why this is the case

        assert model.layer.bias.grad == None
        assert model.layer2.bias.grad == None
        # NOTE: since loss depends only on force and force only on non-constant parts of the NN
        #       resulting in not updating weights during optimisation (would require bias.grad != None)

        torch.autograd.gradcheck(
            self.loss_fn,
            (force[0].double(), grad_ref.double()),
            check_backward_ad=True,
        )
        #################################
        #################################

        print("issue2 -- get gradient of gradient")
        sys.exit(0)
        ##################

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
        dataset, batch_size=10, drop_last=False, shuffle=False, num_workers=8
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
