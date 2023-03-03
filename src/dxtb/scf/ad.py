import torch

from .._types import Any, Tensor
from ..basis import IndexHelper
from ..interaction import Interaction
from .iterator import solve


class SelfConsistentFieldAD(torch.autograd.Function):
    """
    Pytorch autograd class to allow for analytical gradient for SCF calculation.
    """

    @staticmethod
    def forward(
        ctx: "SelfConsistentFieldAD",
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor,
        interactions: Interaction,
        ihelp: IndexHelper,
        guess: str,
        hcore: Tensor,
        overlap: Tensor,
        occupation: Tensor,
        n0: Tensor,
        fwd_options: dict[str, Any],
        scf_options: dict[str, Any],
        use_potential: bool,
    ) -> tuple[Tensor, ...]:
        """Forward pass for evaluation of SCF procedure.

        Parameters
        ----------
        ctx : SelfConsistentFieldAD
            Context object required to stash values to `.backward()`.
            You can cache arbitrary objects for use in the backward pass
            using the ctx.save_for_backward method.
        numbers : Tensor
            Atomic numbers of the system.
        positions : Tensor
            Positions of the system.
        chrg : Tensor
            Total charge.
        interactions : Interaction
            Interaction object.
        ihelp : IndexHelper
            Index helper object.
        guess : str
            Name of the method for the initial charge guess.
        hcore : Tensor
            Core Hamiltonian.
        overlap : Tensor
            Overlap matrix.
        occupation : Tensor
            Occupation of states.
        n0 : Tensor
            Reference occupations for each orbital.
        fwd_options : dict[str, Any]
            Options for xitorch optimisation.
        scf_options : dict[str, Any]
            Options for scf.
        use_potential : bool
            Either iterate potential or charges in self-consistent field procedure.

        Returns
        -------
        tuple[Tensor]
            Collection of scf output, such as energies, charges and hamiltonian.
        """
        # NOTE: *args and **kwargs not available
        result = solve(
            numbers,
            positions,
            chrg,
            interactions,
            ihelp,
            guess,
            hcore,
            overlap,
            occupation,
            n0,
            fwd_options=fwd_options,
            scf_options=scf_options,
            use_potential=use_potential,
        )

        # only tensor objects returned (move to test)
        assert all([isinstance(v, Tensor) for _, v in result.items()])

        # save for backward()
        ctx.interactions = interactions

        return (
            result["charges"],
            result["density"],
            result["emo"],
            result["energy"],
            result["fenergy"],
            result["hamiltonian"],
        )

    @staticmethod
    def backward(ctx, charges, density, emo, energy, fenergy, hamiltonian):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        NOTE:
            * each argument is the gradient w.r.t the given output (grad_output)
            * each returned value should be the gradient w.r.t. the corresponding input
        If an input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a gradient for that input.
        """
        # Further information: https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html
        # Good example: https://github.com/pytorch/pytorch/issues/16940

        # TODO: implement analytical gradient for scf here (via interactions.get_grad())
        grads = ctx.interactions.get_gradient()
        raise NotImplementedError

        # should return as many tensors, as there were inputs to forward()
        return (
            numbers,
            positions,
            chrg,
            interactions,
            ihelp,
            None,
            hcore,
            overlap,
            occupation,
            n0,
            None,
            None,
            None,
        )
