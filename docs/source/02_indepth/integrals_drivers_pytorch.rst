.. _indepth_integrals_drivers_pytorch:

PyTorch
=======

.. automodule:: dxtb._src.integral.driver.pytorch
   :no-index:

Overlap, dipole and quadrupole integrals
----------------------------------------

The PyTorch drivers build the overlap, dipole and quadrupole integrals (and
thereby GFN2-xTB, which needs the multipoles) without libcint. All integrals
are assembled from one-dimensional integrals, which can be computed with one
of two interchangeable algorithms, selected with the ``int_algorithm`` option
(command line: ``--int-algorithm``):

- ``os`` (default): Obara-Saika, three-index vertical recursion
- ``md``: McMurchie-Davidson with Hermite moments

Both algorithms agree with libcint to ``1e-12`` in double precision, support
CPU and CUDA, padded batches, and derivatives of any order (also with
``torch.func``) with respect to the positions and the basis parameters, and
work with ``torch.compile``. The overlap gradient required by the analytical
nuclear gradient is computed from the same one-dimensional integrals.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}
    numbers = torch.tensor([8, 1, 1], device=dd["device"])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], **dd
    )

    opts = {"int_driver": "pytorch", "int_algorithm": "os"}
    calc = dxtb.calculators.GFN2Calculator(numbers, opts=opts, **dd)
    energy = calc.get_energy(positions)

The option is only valid for the PyTorch drivers; it is rejected for libcint.

The legacy driver (``int_driver="legacy"``) is the original loop-based overlap
with explicitly written-down McMurchie-Davidson coefficients. It is kept for
reference, has no overlap gradient, and takes its multipole integrals from the
same algorithms.

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
