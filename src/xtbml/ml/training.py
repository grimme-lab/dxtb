""" Simple training pipline for pytoch ML model. """

import datetime
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd

from xtbml.ml.util import load_model_from_cfg
from xtbml.data.dataset import ReactionDataset


def train():
    """Trains the model."""

    # TODO: load data correctly
    dataset = get_dummy_dataset()

    # TODO: extend padding to number of parameters
    # prune to constant number of partners
    if True:
        np = [len(r.partners) for r in dataset.reactions]
        idxs = [i for i, x in enumerate(np) if x != 2]
        for i in reversed(idxs):
            dataset.rm_reaction(idx=i)
        cc = [r.uid for r in dataset.reactions]
        print(f"Dataset contains {len(cc)} reactions: {cc}")

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 2})

    # set config for ML
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_optimizer": "Adam",
        "training_loss_fn": "L1Loss",
        "training_lr": 0.01,
        "epochs": 100,
    }

    # load components
    model, optimizer, loss_fn, scheduler = load_model_from_cfg(cfg_ml, load_state=False)

    # run training
    model.train()
    losses = []
    mads, mads_ref = [], []
    for i in range(cfg_ml["epochs"]):
        print(f"EPOCH {i}")
        losses_epoch = []
        mads_epoch, mads_ref_epoch = [], []
        for batched_samples, batched_reaction in dl:

            # some samples required in reaction are not available
            if batched_samples == []:
                continue

            # prediction based on QM features
            y = model(batched_samples, batched_reaction)

            # TODO: where to use reaction.egfn1?

            # reshape to avoid broadcasting
            # y = y.view(y.size(0), -1)
            y_true = batched_reaction.eref

            # optimize model parameter and feature parameter
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

        # Loss per epoch is average of batch losses
        losses.append(sum(losses_epoch) / len(losses_epoch))
        mads.append(sum(mads_epoch) / len(mads_epoch))
        mads_ref.append(sum(mads_ref_epoch) / len(mads_ref_epoch))

    print("Finished training")

    df = pd.DataFrame(
        list(zip(losses, mads, mads_ref)), columns=["Losses", "MAD", "MAD_ref"]
    )

    # save model
    uid = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    save_name = f"../models/{uid}"
    df.to_csv(save_name + "_df.csv", index=True)
    torch.save(model.state_dict(), save_name + "_model.pt")


def get_dummy_dataset() -> ReactionDataset:
    """Return a preliminary dataset for setting up the workflow."""

    # load json with real data
    path_samples = Path(Path.cwd(), "../data/features.json")
    path_reactions = Path(Path.cwd(), "../data/reactions.json")

    # samples = Samples.from_json(path_samples)
    # reactions = Reactions.from_json(path_reactions)
    dataset = ReactionDataset.create_from_disk(
        path_reactions=path_reactions, path_samples=path_samples
    )

    # apply simple padding
    dataset.pad()

    return dataset
