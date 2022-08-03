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

    if True:

        # calculate dummy gradient, i.e. null gradient, as
        # fwADs requires an existing gradient to convert to dual tensors

        # factor for backward propagation, i.e. dQ/dQ = external_gradient
        external_gradient = torch.zeros_like(sample.scf)
        sample.scf.backward(gradient=external_gradient, retain_graph=True)

        print("sample.scf", sample.scf)

        # gradient of SCF
        print(sample.a.grad)
        print(sample.b.grad)

        # for: external_gradient = torch.ones_like(sample.scf)
        """assert torch.all(9 * sample.a**2 == sample.a.grad)
        assert torch.all(-2 * sample.b == sample.b.grad)"""

    # 4. calculate ML
    model = ML()

    # references (arbitrary)
    grad_ref = torch.tensor([35.0, 80.0])
    e_ref = torch.tensor([425.5])

    loss_fn = nn.MSELoss(reduction="mean")

    with fwAD.dual_level():
        # NOTE: existing tensors only converted to dual number if
        #       previously gradient calculated, i.e. via .backward().

        # since development in progress, directly setting dual parameters not possible
        # temporary workaround: https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
        params = {name: p for name, p in model.named_parameters()}
        tangents = {name: torch.ones_like(p) for name, p in params.items()}
        # tangents define the dependency on input variables, i.e. scale the gradient

        for name, p in params.items():
            rdelattr(model, name)
            rsetattr(model, name, fwAD.make_dual(p, tangents[name]))

        # NOTE: model.named_parameters() is empty afterwards
        assert len(list(model.named_parameters())) == 0

        # 4.2 calculate energies
        energies = model(sample)
        # QUESTION: does that set all ml.params.require_grad == TRUE?
        # QUESTION / TODO: require to turn off require grad to avoid leave nodes (e.g. biases) to store gradient for force calc
        # with torch.nograd(): ...
        assert energies.equal(torch.tensor([435.5]))

        print("sample.scf", sample.scf)
        print("energies", energies)

        # real part of dual number
        primal = fwAD.unpack_dual(energies).primal
        # jacobian vector product
        jvp = fwAD.unpack_dual(energies).tangent

        print("before")
        print(sample.a.grad)  # no gradient (Problem!)
        print(type(sample.a.grad))

        # 5. calculate force based on forward graph
        energies.backward()
        print("after")
        print(sample.a.grad)  # this has a gradient (tangent)!
        print(sample.b.grad)
        print(type(sample.a.grad))

        # for fixed layer parameter (checked via calculus by hand)
        assert sample.a.grad.equal(torch.tensor([252.0, 648.0]))
        assert sample.b.grad.equal(torch.tensor([-84.0, -64.0]))

        # TODO: cross check these values
        assert fwAD.unpack_dual(sample.a.grad).tangent.equal(torch.tensor([36.0, 81.0]))
        assert fwAD.unpack_dual(sample.b.grad).tangent.equal(
            torch.tensor([-12.0, -8.0])
        )

        # 6. calculate loss and update model parameters
        loss = loss_fn(sample.a.grad, grad_ref)
        print("loss", loss)
        print("loss.has_grad", loss.requires_grad)
        gg = fwAD.unpack_dual(loss).tangent
        print(
            gg, gg.requires_grad
        )  # <-- has no .grad (PROBLEM!) -- but now at least a tangent

        # loss.backward()  # <-- does not accept the tangent as gradient!
        # TODO: write custom autograd function for this (does the backward argument require .requires_grad?)

    print("has loss a gradient now?")
    return

    # NOTE: also see:
    # * https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
    # * https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
    # * https://pytorch.org/functorch/stable/generated/functorch.grad.html
    # * jax cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff

    ############


if __name__ == "__main__":
    main()
    # custom_autograd()
