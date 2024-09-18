.. _quickstart-getting-started:

Getting Started
===============

The main object of *dxtb* is the :class:`~dxtb.Calculator` class, which is used
to perform calculations on a given system.

Note that all quantities are in atomic units.

Creating a Calculator
---------------------

The constructor always requires the atomic numbers of the system(s) and a
tight-binding parametrization.
Currently, we provide the :data:`~dxtb.GFN1_XTB` parametrization out of the box.
If you directly use the corresponding
:class:`~dxtb.calculators.GFN1Calculator`, only the atomic numbers are required.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])  # LiH
    calc = dxtb.calculators.GFN1Calculator(numbers)

We recommend to always pass the (floating point) :class:`~torch.dtype` and
:class:`~torch.device` arguments to the constructor to ensure consistency.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

Using the Calculator
--------------------

Now, you can request various properties of the system. The most common one is
the total energy.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    energy = calc.energy(positions)

    print(energy)

If your system is charged or has unpaired electrons, you need to supply both
quantities as optional keyword arguments to :meth:`~dxtb.Calculator.energy`.

.. code-block:: python

    energy = calc.energy(positions, charge=0, spin=0)

Instead of calling the :meth:`~dxtb.Calculator.energy` method, you can also
use corresponding getters :meth:`~dxtb.Calculator.get_energy`:

.. code-block:: python

    energy = calc.get_energy(positions, charge=0, spin=0)

We recommend using the getters, as they provide the familiar ASE-like interface.


Gradients
---------

To calculate the gradients of the energy with respect to the atomic positions,
you can use the standard :func:`torch.autograd.grad` function.
Remember to set the ``requires_grad`` attribute of the positions tensor to
``True``.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions.requires_grad_(True)

    energy = calc.energy(positions)
    (g,) = torch.autograd.grad(energy, positions)

    print(g)

For convenience, you can use the :meth:`~dxtb.Calculator.forces` or
:meth:`~dxtb.Calculator.get_forces` method directly.

.. code-block:: python

      forces = calc.forces(positions)
      forces = calc.get_forces(positions)

The equivalency of the two methods (except for the sign) can be verified by
the example `here <https://github.com/grimme-lab/dxtb/blob/main/examples/forces.py>`_.


.. warning::

    If you supply the **same inputs** to the calculator multiple times with
    gradient tracking enabled, you have to reset the calculator in between with
    :meth:`~dxtb.Calculator.reset_all`. Otherwise, the gradients will be wrong.

    .. admonition:: Example
       :class: toggle

       .. code-block:: python

           import torch
           import dxtb

           dd = {"dtype": torch.double, "device": torch.device("cpu")}

           numbers = torch.tensor([3, 1], device=dd["device"])
           positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

           calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

           pos = positions.clone().requires_grad_(True)
           energy = calc.energy(pos)
           (g1,) = torch.autograd.grad(energy, pos)

           # wrong gradients without reset here
           calc.reset_all()

           pos = positions.clone().requires_grad_(True)
           energy = calc.energy(pos)
           (g2,) = torch.autograd.grad(energy, pos)

           assert torch.allclose(g1, g2)


More Properties
---------------

Besides :meth:`~dxtb.Calculator.get_energy` / :meth:`~dxtb.Calculator.energy`
and :meth:`~dxtb.Calculator.get_forces` / :meth:`~dxtb.Calculator.forces`,
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

Each method has a corresponding getter and some additional properties are also
accessible via getters:

- :meth:`~dxtb.Calculator.get_normal_modes`: Normal modes from vibrational analysis.
- :meth:`~dxtb.Calculator.get_frequencies`: Vibrational frequencies.
- :meth:`~dxtb.Calculator.get_ir_intensities`: Infrared intensities.
- :meth:`~dxtb.Calculator.get_raman_intensities`: Raman intensities.
- :meth:`~dxtb.Calculator.get_raman_depol`: Raman depolarization ratios.
- :meth:`~dxtb.Calculator.get_charges` /
  :meth:`~dxtb.Calculator.get_mulliken_charges`: Mulliken charges from SCF.
- :meth:`~dxtb.Calculator.get_iterations`: Number of SCF iterations.

Note that all methods (except :meth:`~dxtb.Calculator.energy`) utilize
automatic derivatives. For comparison, each method also has a numerical
counterpart, e.g., :meth:`~dxtb.Calculator.forces_numerical`.

.. note:: Caching

    These methods only calculate the requested property. To also store
    associated properties, turn on caching by passing
    ``{"cache_enabled": True}`` to the calculator options. This avoids
    redundant calculations. For example, with caching,
    :meth:`~dxtb.Calculator.get_hessian` also stores the forces and the energy.
    Hence, a subsequent :meth:`~dxtb.Calculator.get_forces` does not
    necessitate an additional calculation.

For more details, please see the :ref:`here <indepth_calculators>`.
