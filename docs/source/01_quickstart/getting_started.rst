.. _quickstart-getting-started:

Getting Started
===============

The main object of *dxtb* is the :class:`~dxtb.Calculator` class, which is used
to perform calculations on a given system.

Creating a Calculator
---------------------

The constructor always requires the atomic numbers of the system(s) and a
tight-binding parametrization.
Currently, we provide the :data:`~dxtb.GFN1_XTB` parametrization out of the box.
If you directly use the corresponding :class:`~dxtb.GFN1Calculator`, only the
atomic numbers are required.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])  # LiH
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
quantities as optional keyword arguments to :meth:`energy`.

.. code-block:: python

    energy = calc.energy(positions, charge=0, spin=0)


Gradients
---------

To calculate the gradients of the energy with respect to the atomic positions,
you can use the standard :func:`torch.autograd.grad` function.
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

For convenience, you can use the :meth:`forces` method directly.

.. code-block:: python

      forces = calc.forces(positions)

The equivalency of the two methods (except for the sign) can be verified by
the example `here <https://github.com/grimme-lab/dxtb/blob/main/examples/forces.py>`_.


More Properties
---------------

Besides :meth:`~dxtb.Calculator.energy` and :meth:`~dxtb.Calculator.forces`,
the :class:`~dxtb.Calculator` class provides methods to calculate various other
quantities. The full list is given below:

- :meth:`~dxtb.Calculator.energy`: Total energy.
- :meth:`~dxtb.Calculator.forces`: Nuclear forces (negative gradient).
- :meth:`~dxtb.Calculator.dipole`: Electric dipole moment.
- :meth:`~dxtb.Calculator.dipole_deriv`: Derivative of electric dipole moment w.r.t. nuclear positions.
- :meth:`~dxtb.Calculator.polarizability`: Electric dipole polarizability.
- :meth:`~dxtb.Calculator.pol_deriv`: Derivative of electric dipole polarizability w.r.t. nuclear positions.
- :meth:`~dxtb.Calculator.hyperpolarizability`: Electric hyperpolarizability.
- :meth:`~dxtb.Calculator.hessian`: Hessian matrix.
- :meth:`~dxtb.Calculator.vibration`: Vibrational frequencies and normal modes.
- :meth:`~dxtb.Calculator.ir`: Infrared intensities.
- :meth:`~dxtb.Calculator.raman`: Raman intensities.

Note that all methods (except :meth:`~dxtb.Calculator.energy`) utilize
automatic derivatives. For comparison, each method also has a numerical
counterpart, e.g., :meth:`~dxtb.Calculator.forces_numerical`.

For more details, please see the :ref:`here <indepth_calculators>`.
