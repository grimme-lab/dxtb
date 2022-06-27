""" Simple training pipline for pytoch ML model. """


import datetime
import torch
import pandas as pd
from pathlib import Path

from ..ml.util import load_model_from_cfg
from ..data.dataset import get_gmtkn_dataset
from .norm import Normalisation
from .loss import WTMAD2Loss
from ..data.adjacency import calc_adj


def train():
    """Trains the model."""

    # load GMTKN55 from disk
    root = Path(__file__).resolve().parents[3]
    dataset = get_gmtkn_dataset(Path(root, "data"))

    # optional pruning
    # dataset = dataset[:50]  # [4:6]
    print("len(dataset)", len(dataset))

    # calculate adjacency matrix (GNN only)
    for s in dataset.samples:
        s.adj = calc_adj(s).type(s.dtype)

    # TODO: add feature normalisation

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 5, "shuffle": True})

    # set config for ML
    cfg_ml = {
        "model_architecture": "GNN",  # "Basic_CNN" "EGNN" "Simple_Net" "GNN"
        "training_optimizer": "Adam",
        "training_scheduler": "CyclicLR",
        "training_loss_fn": "WTMAD2Loss",
        "training_loss_fn_path": Path(root, "data"),
        "training_lr": 0.01,
        "epochs": 60,
    }

    # load components
    model, optimizer, loss_fn, scheduler = load_model_from_cfg(cfg_ml, load_state=False)

    bookkeeping = []
    losses, gfn1_losses, gfn2_losses = [], [], []

    # run training
    model.train()
    for epoch in range(cfg_ml["epochs"]):
        print(f"EPOCH {epoch}")

        losses_epoch, gfn1_loss_epoch, gfn2_loss_epoch = [], [], []

        for batched_samples, batched_reaction in dl:

            # TODO: make sure that all tensors require grad!

            # TODO: ensure that added PADDING sample in batched_samples do not change prediction OR loss result (e.g. the mean or sum)
            # --> mask all padded values --> check that especially with the ML-model

            # TODO: GNN
            #       ensure correct batching of adj

            # prediction based on QM features
            y = model(batched_samples, batched_reaction)
            y_true = batched_reaction.eref
            # TODO: where to use reaction.egfn1?

            # optimize model parameter and feature parameter
            if isinstance(loss_fn, WTMAD2Loss):
                # derive subset from partner list
                subsets = [s.split("/")[0] for s in batched_reaction.partners]
                # different number of partners per reaction
                n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)

                loss = loss_fn(y, y_true, subsets, n_partner)
            else:
                loss = loss_fn(y, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"loss: {loss} | y: {y} | y_true: {y_true}")

            # bookkeeping
            if isinstance(loss_fn, WTMAD2Loss):
                gfn1_loss = loss_fn(batched_reaction.egfn1, y_true, subsets, n_partner)
                gfn2_loss = loss_fn(batched_reaction.egfn2, y_true, subsets, n_partner)
            else:
                gfn1_loss = loss_fn(batched_reaction.egfn1, y_true)
                gfn2_loss = loss_fn(batched_reaction.egfn2, y_true)

            bookkeeping.append(
                {
                    "epoch": epoch,
                    "y": y,
                    "y_true": y_true,
                    "loss": loss.item(),
                    "gfn1_loss": gfn1_loss.item(),
                    "gfn2_loss": gfn2_loss.item(),
                    "egfn1": batched_reaction.egfn1,
                    "egfn2": batched_reaction.egfn2,
                }
            )
            losses_epoch.append(loss.item())
            gfn1_loss_epoch.append(gfn1_loss.item())
            gfn2_loss_epoch.append(gfn2_loss.item())

        # adapting learning rate epoch-wise
        scheduler.step()

        # Loss per epoch is average of batch losses
        losses.append(sum(losses_epoch) / len(losses_epoch))
        gfn1_losses.append(sum(gfn1_loss_epoch) / len(gfn1_loss_epoch))
        gfn2_losses.append(sum(gfn2_loss_epoch) / len(gfn2_loss_epoch))
        print(f"\t loss: ", losses[-1])
        print(f"\t GFN1 \t {gfn1_losses[-1]}")
        print(f"\t GFN2 \t {gfn2_losses[-1]}")

    print("Minimal Loss:", min(losses))

    # save model
    uid = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    save_name = f"../models/{uid}"
    df = pd.DataFrame(
        list(zip(losses, gfn1_losses, gfn2_losses)),
        columns=["Losses", "GFN1", "GFN2"],
    )
    df.to_csv(save_name + "_df.csv", index=True)
    pd.DataFrame(bookkeeping).to_csv(save_name + "_bookkeeping.csv", index=True)
    torch.save(model.state_dict(), save_name + "_model.pt")
    with open(save_name + "_cfg.txt", "w") as f:
        cfg_ml["data_root"] = str(root)
        cfg_ml["data_len"] = len(dataset)
        f.write(str(cfg_ml))

    print(f"Model saved as {save_name}")
    print("Finished training")
