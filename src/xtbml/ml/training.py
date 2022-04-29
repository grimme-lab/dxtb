""" Simple training pipline for pytoch ML model. """

import torch
import json

from xtbml.ml.util import load_model_from_cfg
from xtbml.data.dataset import Sample, Reaction, ReactionDataset


def train():
    """Trains the model."""

    # TODO: load data correctly
    dataset = get_dummy_dataset()

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 2})

    # set config for ML
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_optimizer": "Adam",
        "training_loss_fn": "MSELoss",
        "training_lr": 0.01,
        "epochs": 3,
    }

    # load components
    model, optimizer, loss_fn, scheduler = load_model_from_cfg(cfg_ml, load_state=False)

    # run training
    model.train()
    for i in range(cfg_ml["epochs"]):
        print(f"EPOCH {i}")
        for (batched_samples, batched_reaction) in dl:

            # prediction based on QM features
            y = model(batched_samples, batched_reaction)

            # TODO: where to use reaction.egfn1?

            # reshape to avoid broadcasting
            y = y.view(y.size(0), -1)
            y_true = batched_reaction.eref

            # optimize model parameter and feature parameter
            loss = loss_fn(y, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: optimise parameters back to initial positions
            #       (propagate gradients and update parameters)

    print("Finished training")


def get_dummy_dataset() -> ReactionDataset:
    """Return a preliminary dataset for setting up the workflow."""
    # load json with real data
    with open("/home/ch/PhD/projects/xtbML/testfeatures.json", "r") as f:
        data = json.load(f)

    dummy_samples = []
    for k, v in data.items():
        abc = Sample(
            uid=k,
            xyz=torch.tensor(v["xyz"]),
            numbers=torch.tensor(v["numbers"]),
            egfn1=torch.tensor(v["egfn1"]),
            ovlp=torch.tensor(v["ovlp"]),
            h0=torch.tensor(v["h0"]),
            cn=torch.tensor(v["cn"]),
        )
        dummy_samples.append(abc)

    r = Reaction(
        uid="TEST",
        partners=["MB16-43/LiH", "MB16-43/42"],
        nu=torch.tensor([-1, 1]),
        egfn1=torch.tensor([1.23]),
        eref=torch.tensor([1.54]),
    )

    dummy_reactions = [r, r]
    dataset = ReactionDataset(samples=dummy_samples, reactions=dummy_reactions)

    # apply simple padding
    dataset.pad()

    return dataset
