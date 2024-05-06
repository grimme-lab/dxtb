.. _quickstart-getting-started:

Getting Started
===============

The main object of *dxtb* is the ``Calculator`` class, which is used to perform
calculations on a given system.

Creating a Calculator
---------------------

The constructor always requires the atomic numbers of the system(s) and a
tight-binding parametrization.
Currently, we provide the GFN1-xTB parametrization out of the box.
If you directly use the corresponding Calculator, only the atomic numbers are
required.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])
    calc = dxtb.GFN1Calculator(numbers)

We recommend to always pass the (floating point) ``dtype`` and the
``device`` arguments to the constructor to ensure consistency.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.GFN1Calculator(numbers, **dd)

Using the Calculator
--------------------

Now, you can request various properties of the system. The most common one is
the total energy.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    energy = calc.energy(positions)

    print(energy)

If your system is charged or has unpaired electrons, you need to supply both
quantities as optional keyword arguments to the ``energy`` method.

.. code-block:: python

    energy = calc.energy(positions, charge=0, spin=0)


Gradients
---------

To calculate the gradients of the energy with respect to the atomic positions,
you can use the standard ``torch.autograd.grad`` function.
However, you need to set the ``requires_grad`` attribute of the positions tensor
to ``True``.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions.requires_grad_(True)

    energy = calc.energy(positions)
    (g,) = torch.autograd.grad(energy, positions)

    print(g)
