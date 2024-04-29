.. _quickstart_examples:

Simple Examples
===============

The main object of *dxtb* is the `Calculator` class, which is used to perform
calculations on a given system.

Instantiation
-------------

The constructor always requires the atomic numbers of the system(s) and a
tight-binding parametrization. Currently, we provide the GFN1-xTB
parametrization out of the box.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB)

We recommend to always pass the (floating point) `dtype` and the
`device` arguments to the constructor to ensure consistency.

.. code-block:: python

    import torch
    import dxtb
    from dxtb.typing import DD

    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)

Using the Calculator
--------------------

Now, you can request various properties of the system. The most common one is
the total energy.

.. code-block:: python

    import torch
    import dxtb
    from dxtb.typing import DD

    dd: DD = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    energy = calc.energy(numbers, positions)

    print(energy)

If your system is charged or has unpaired electrons, you need to supply both
quantities as optional keyword arguments to the `energy` method.

.. code-block:: python

    energy = calc.energy(numbers, positions, charge=0, spin=0)
