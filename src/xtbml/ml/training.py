""" Simple training pipline for pytoch ML model. """

import datetime
import imp
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import pandas as pd

from ..ml.util import load_model_from_cfg
from ..data.dataset import get_gmtkn_dataset
from .norm import Normalisation


def train():
    """Trains the model."""

    # load GMTKN55 from disk
    root = Path(__file__).resolve().parents[3]
    dataset = get_gmtkn_dataset(Path(root, "data"))

    # TODO: maybe wrap this into a transforms module
    #   (i.e. module that act on dataset objects)
    Normalisation.normalise(dataset)

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 2, "shuffle": True})

    # set config for ML
    cfg_ml = {
        "model_architecture": "Basic_CNN",  # "Basic_CNN" "EGNN"
        "training_optimizer": "Adam",
        "training_scheduler": "CyclicLR",
        "training_loss_fn": "WTMAD2Loss",  # "L1Loss",
        "training_lr": 0.01,  # 0.0001 0.001 0.01
        "epochs": 60,
    }

    # load components
    model, optimizer, loss_fn, scheduler = load_model_from_cfg(
        root, cfg_ml, load_state=False
    )

    # run training
    model.train()
    losses = []
    mads, mads_ref, wtm2_ref = [], [], []
    for i in range(cfg_ml["epochs"]):
        print(f"EPOCH {i}")
        losses_epoch = []
        mads_epoch, mads_ref_epoch, wtm2_ref_epoch = [], [], []
        for batched_samples, batched_reaction in dl:

            # TODO: make sure that all tensors require grad!

            # TODO: ensure that added PADDING sample in batched_samples do not change prediction OR loss result (e.g. the mean or sum)
            # --> mask all padded values --> check that especially with the ML-model

            # prediction based on QM features
            y = model(batched_samples, batched_reaction)

            # TODO: where to use reaction.egfn1?

            # reshape to avoid broadcasting
            # y = y.view(y.size(0), -1)
            y_true = batched_reaction.eref

            # optimize model parameter and feature parameter
            if cfg_ml["training_loss_fn"] == "WTMAD2Loss":
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
            # TODO: optimise parameters back to initial positions
            #       (propagate gradients and update parameters)

            # print(f"loss: {loss} | y: {y} | y_true: {y_true}")

            # bookkeeping
            mad_fn = nn.L1Loss(reduction="mean")
            losses_epoch.append(loss.item())
            mads_epoch.append(mad_fn(y, y_true).item())
            mads_ref_epoch.append(mad_fn(batched_reaction.egfn1, y_true).item())

            if cfg_ml["training_loss_fn"] == "WTMAD2Loss":
                wtm2_ref_epoch.append(
                    loss_fn(batched_reaction.egfn1, y_true, subsets, n_partner).item()
                )

        # adapting learning rate epoch-wise
        scheduler.step()

        # Loss per epoch is average of batch losses
        losses.append(sum(losses_epoch) / len(losses_epoch))
        mads.append(sum(mads_epoch) / len(mads_epoch))
        mads_ref.append(sum(mads_ref_epoch) / len(mads_ref_epoch))
        wtm2_ref.append(sum(wtm2_ref_epoch) / len(wtm2_ref_epoch))
        print(
            f"Loss: {losses[-1]} | MAD: {mads[-1]} | MAD_REF: {mads_ref[-1]} | WTM2_REF: {wtm2_ref[-1]}"
        )

    print("Minimal Loss:", min(losses))

    print("Finished training")

    df = pd.DataFrame(
        list(zip(losses, mads, mads_ref, wtm2_ref)),
        columns=["Losses", "MAD", "MAD_ref", "WTMAD2_ref"],
    )

    # save model
    uid = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    save_name = f"../models/{uid}"
    df.to_csv(save_name + "_df.csv", index=True)
    torch.save(model.state_dict(), save_name + "_model.pt")
