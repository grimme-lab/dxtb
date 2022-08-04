import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import time
import torch.autograd.forward_ad as fwAD
import torchviz
import functools
from xtbml.typing import Tensor


# TODO: check that for model=Identity() the force gradient is identical to scf force gradient (e.g. from tblite)


""" Helper function for accesing nested object properties. """


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rdelattr(obj, attr):
    pre, _, post = attr.rpartition(".")
    return delattr(rgetattr(obj, pre) if pre else obj, post)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class Sample:

    pos: Tensor
    energies: Tensor
    pop: Tensor
    scf: Tensor

    a: Tensor  # e.g. positions
    b: Tensor


def scf(sample):

    """charges = torch.arange(sample.pos.shape[0]).float()

    # simple calculation including sample position
    scf_feature = sample.pos @ charges"""

    # easier case
    scf_feature = 3 * sample.a**3 - sample.b**2

    return {"charges": None, "scf_feature": scf_feature}


class ML(nn.Module):
    # short: model = nn.Linear(2, 1)
    def __init__(self):
        super().__init__()

        self.layer = torch.nn.Linear(2, 1)

        # fix layer parameter
        self.layer.weight = torch.nn.Parameter(torch.tensor([[7.0, 8.0]]))
        self.layer.bias = torch.nn.Parameter(torch.tensor([-0.5]))

    def forward(self, sample):
        x = self.layer(sample.scf)
        return x


def main():

    print("simple test scenario for checking correct gradients")

    # 1. get sample and set requires_grad
    sample = Sample()
    # sample.pos = torch.arange(9.0, requires_grad=True).reshape([3, 3])
    # print(sample.pos)
    # assert sample.pos.requires_grad
    sample.a = torch.tensor([2.0, 3.0], requires_grad=True)
    sample.b = torch.tensor([6.0, 4.0], requires_grad=True)
    assert sample.a.requires_grad

    # 2. calculate SCF
    results = scf(sample)

    # 3. update sample
    sample.scf = results["scf_feature"]

    # 4. calculate ML
    model = ML()

    # references (arbitrary)
    grad_ref = torch.tensor([35.0, 80.0])
    e_ref = torch.tensor([425.5])

    loss_fn = nn.MSELoss(reduction="mean")

    # optimising based on energies (classical)
    if False:
        energies = model(sample)
        loss = loss_fn(energies, e_ref)
        assert loss.equal(torch.tensor(100.0))
        assert loss.requires_grad

        print(model.layer.weight)
        print(model.layer.bias)
        print(model.layer.weight.grad)
        print(model.layer.bias.grad)

        # gradient to NN parameter
        loss.backward()

        print(model.layer.weight)
        print(model.layer.bias)
        print(model.layer.weight.grad)  # gradient propagated
        print(model.layer.bias.grad)

        assert model.layer.weight.grad != None
        assert model.layer.bias.grad != None

        return

    # optimising based on forces (novel)

    # 4.2 calculate energies
    energies = model(sample)
    # QUESTION: does that set all ml.params.require_grad == TRUE?
    # QUESTION / TODO: require to turn off require grad to avoid leave nodes (e.g. biases) to store gradient for force calc
    # with torch.nograd(): ...
    assert energies.equal(torch.tensor([435.5]))
    print("energies", energies)

    # 5. calculate force based on forward graph
    force = torch.autograd.grad(
        energies,
        sample.a,
        grad_outputs=torch.ones_like(energies),
        # retain_graph=True,
        create_graph=True,
    )
    # NOTE: in real application set opt.zero_grad() before
    print("force", force)  # this has a gradient!
    print(force[0].shape, grad_ref.shape)
    if False:
        # alternatively propagate gradient until leave nodes
        #   however sample.a.grad has no grad. Via dual-tensors
        #   a jvp tangent can be calculated (cf. forward AD).
        energies.backward()
        # for fixed layer parameter (checked via calculus by hand)
        assert sample.a.grad.equal(torch.tensor([252.0, 648.0]))
        assert sample.b.grad.equal(torch.tensor([-84.0, -64.0]))

    # 6. calculate loss and update model parameters
    loss = loss_fn(force[0], grad_ref)  # loss tensor(184856.5000)
    # also see: https://stackoverflow.com/questions/71294401/pytorch-loss-function-that-depends-on-gradient-of-network-with-respect-to-input

    print("loss", loss)  # loss tensor(184856.5000, tangent=53820.0)
    print("loss.has_grad", loss.requires_grad)

    print("before")
    print(sample.a.grad)
    print(sample.b.grad)
    print(model.layer.weight.grad)
    print(model.layer.bias.grad)
    # ensure that loss is not backpropagated
    # until positions but only until NN parameters
    sample.a.requires_grad_(False)
    sample.b.requires_grad_(False)

    # propagate gradients for updating model parameters
    loss.backward()
    print("after")
    print(sample.a.grad)
    print(sample.b.grad)
    print(model.layer.weight)
    print(model.layer.bias)
    print(model.layer.weight.grad)
    print(model.layer.bias.grad)

    assert model.layer.weight.grad != None  # tensor([[ 7812., 46008.]])
    # NOTE: does not change, regardless whether sample.requires_grad(True/False)
    assert model.layer.bias.grad == None  # TODO: why different from energy-loss?

    # loss needs a gradient!
    print("does autograd solve the gradient issue?")
    return

    ############
    # NOTE: also see:
    # * https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
    # * https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
    # * https://pytorch.org/functorch/stable/generated/functorch.grad.html
    # * jax cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff
    # * https://stackoverflow.com/questions/71294401/pytorch-loss-function-that-depends-on-gradient-of-network-with-respect-to-input
    ############


if __name__ == "__main__":
    main()
