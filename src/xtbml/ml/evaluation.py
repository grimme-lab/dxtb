from pathlib import Path

import pandas as pd
import torch

from ..data.adjacency import calc_adj
from ..data.dataset import create_subset, get_gmtkn55_dataset
from .loss import WTMAD2Loss
from .util import load_model_from_cfg

""" Load ML model from disk and conduct basic evaluation. """


def evaluate():
    """Simple evaluation of performance of a given model."""

    root = Path(__file__).resolve().parents[3]
    torch.set_printoptions(precision=2, sci_mode=False)

    # load data
    dataset = get_gmtkn55_dataset(Path(root, "data"))
    create_subset(dataset, ["barrier"])  # OPTIONAL: select subset(s)
    # barrier thermo_small thermo_large nci_inter nci_intra
    # dataset = dataset[:50] # OPTIONAL: pruning of dataset
    print("len(dataset)", len(dataset))

    # calculate adjacency matrix (GNN only)
    for s in dataset.samples:
        s.adj = calc_adj(s).type(s.dtype)

    # bookkeeping
    bset = [r.uid.split("_")[0] for r in dataset.reactions]
    # cc = [r.uid for r in dataset.reactions]
    # print(f"Dataset contains {len(cc)} reactions: {cc}")

    # remove all samples with missing values
    idxs = [i for i in range(len(dataset)) if len(dataset[i][0]) == 0]
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 1})

    # load model
    model_id = "202206271832_cnn_thermoS"  # "202206271605_gnn_gmtkn55" 202206271718_cnn_gmtkn55 202206271804_gnn_thermoS
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_loss_fn_path": Path(root, "data"),
        "training_lr": 0.01,
        "model_state_dict": torch.load(f"{root}/models/{model_id}_model.pt"),
    }
    model, _, _, _ = load_model_from_cfg(cfg_ml)
    model.eval()

    # evaluate model
    eref, egfn1, egfn2, enn = [], [], [], []
    loss_fn = WTMAD2Loss(Path(root, "data"), reduction="mean")
    losses_epoch, gfn1_loss_epoch, gfn2_loss_epoch = [], [], []
    for i, (batched_samples, batched_reaction) in enumerate(dl):

        y = model(batched_samples, batched_reaction)
        y_true = batched_reaction.eref

        eref.append(y_true.item())
        egfn1.append(batched_reaction.egfn1.item())
        egfn2.append(batched_reaction.egfn2.item())
        enn.append(y.item())

        # print(
        #    f"Enn: {enn[i]:.2f} | Eref: {eref[i]:.2f} | Egfn1: {egfn1[i]:.2f} | Samples: {batched_reaction.partners}"
        # )

        subsets = [s.split("/")[0] for s in batched_reaction.partners]
        n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)
        loss = loss_fn(y, y_true, subsets, n_partner)
        gfn1_loss = loss_fn(batched_reaction.egfn1, y_true, subsets, n_partner)
        gfn2_loss = loss_fn(batched_reaction.egfn2, y_true, subsets, n_partner)
        losses_epoch.append(loss.item())
        gfn1_loss_epoch.append(gfn1_loss.item())
        gfn2_loss_epoch.append(gfn2_loss.item())

    df = pd.DataFrame(
        list(zip(bset, eref, egfn1, egfn2, enn)),
        columns=["subset", "Eref", "Egfn1", "Egfn2", "Enn"],
    )
    df["dEgfn1"] = (df["Eref"] - df["Egfn1"]).abs()
    df["dEnn"] = (df["Eref"] - df["Enn"]).abs()
    print(df)
    print(df[["dEgfn1", "dEnn"]].describe())
    print("")
    df.to_csv(f"../models/{model_id}_eval.csv")

    loss_nn = sum(losses_epoch) / len(losses_epoch)
    loss_gfn1 = sum(gfn1_loss_epoch) / len(gfn1_loss_epoch)
    loss_gfn2 = sum(gfn2_loss_epoch) / len(gfn2_loss_epoch)
    print(f"{loss_nn:.2f} | GFN1: {loss_gfn1:.2f} | GFN2: {loss_gfn2:.2f}")

    return bset, eref, egfn1, enn
