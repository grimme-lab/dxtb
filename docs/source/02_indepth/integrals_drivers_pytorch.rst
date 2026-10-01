.. _indepth_integrals_drivers_pytorch:

PyTorch
=======

.. automodule:: dxtb._src.integral.driver.pytorch
   :no-index:

Overlap, dipole and quadrupole integrals
----------------------------------------

The PyTorch drivers build the overlap, dipole and quadrupole integrals (and
thereby GFN2-xTB, which needs the multipoles) without libcint. The
one-dimensional multipole integrals can be computed with one of two
interchangeable algorithms, selected with the ``int_algorithm`` option
(command line: ``--int-algorithm``):

- ``os`` (default): Obara-Saika, three-index vertical recursion
- ``md``: McMurchie-Davidson with Hermite moments

The overlap always uses the explicit McMurchie-Davidson implementation of the
driver. Both algorithms agree with libcint to ``1e-12`` in double precision, support
CPU and CUDA, padded batches, and derivatives of any order (also with
``torch.func``), and work with ``torch.compile``.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}
    numbers = torch.tensor([8, 1, 1], device=dd["device"])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], **dd
    )

    opts = {"int_driver": "analytical", "int_algorithm": "os"}
    calc = dxtb.calculators.GFN2Calculator(numbers, opts=opts, **dd)
    energy = calc.get_energy(positions)

The option is only valid for the PyTorch drivers; it is rejected for libcint.

Parameter gradients (with respect to the basis exponents and contraction
coefficients) are supported by both algorithms. With ``int_driver="analytical"``
the analytical position derivative of the overlap is used unless a basis
parameter requires a gradient, in which case the plain autograd path is taken.

On the CPU, libcint is faster than the PyTorch drivers, in particular for the
multipole integrals of large systems.

Drivers
-------

.. automodule:: dxtb._src.integral.driver.pytorch.driver
   :members:
   :show-inheritance:

Implementations
---------------

.. automodule:: dxtb._src.integral.driver.pytorch.overlap
   :members:
   :show-inheritance:
