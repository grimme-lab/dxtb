import torch


class MyCube(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        result = x**3
        # In regular PyTorch, if we had just run y = x ** 3, then the backward
        # pass computes dx = 3 * x ** 2. In this autograd.Function, we've done
        # that computation here in the forward pass instead.
        dx = 3 * x**2
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        # In order for the autograd.Function to work with higher-order
        # gradients, we must add the gradient contribution of `dx`.
        result = grad_output * dx + grad_dx * 6 * x
        return result


def my_cube(x):
    result, _ = MyCube.apply(x)  # type: ignore
    return result


x = torch.tensor([4, 2], dtype=torch.double)
ggx = torch.func.jacrev(torch.func.jacrev(my_cube))(x)
print(ggx)
print(ggx.sum(0))

from tad_mctc.autograd import hessian

print(hessian(my_cube, (x,), argnums=0))
