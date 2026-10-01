.. _quickstart-spin:

Spin and Unrestricted Calculations
==================================

*dxtb* separates the spin state, the wavefunction mode, and the
spin-polarization energy model:

``spin``
    The number of unpaired electrons, :math:`N_\alpha-N_\beta`.  It controls
    alpha/beta occupation numbers.  It does not by itself enable an
    unrestricted calculation or add an energy interaction.

``uhf_mode``
    A boolean calculator option that selects two independently propagated
    alpha and beta wavefunction channels.  This is a valid mode even when no
    spin-polarization interaction is present.

:class:`~dxtb.components.spin.SpinPolarization`
    A separate, self-consistent, on-site energy interaction.  It acts on the
    shell magnetization charges and contributes a magnetization potential.
    Because that requires alpha and beta wavefunctions, adding the interaction
    automatically enables UHF.

The four possible activation combinations are therefore:

.. list-table:: Activation matrix
   :header-rows: 1

   * - ``uhf_mode``
     - Spin interaction
     - ``nspin``
     - Behavior
   * - ``False``
     - absent
     - 1
     - Restricted; existing behavior and result shapes.
   * - ``True``
     - absent
     - 2
     - Independent UHF; no spin-polarization energy.
   * - ``False``
     - present
     - 2
     - Spin-polarized UHF, promoted by the interaction.
   * - ``True``
     - present
     - 2
     - Explicit UHF with the spin-polarization interaction.

Python quick start
------------------

The following complete example runs the three distinct modes for open-shell
OH.  All quantities are in atomic units.

.. literalinclude:: ../../../examples/spin.py
   :language: python
   :linenos:

The spin-polarization factory is public and can be used with either GFN1-xTB
or GFN2-xTB:

.. code-block:: python

    from dxtb.components.spin import (
        SpinPolarization,
        new_spin_polarization,
    )

    spinpol = new_spin_polarization(numbers, wscale=1.0, **dd)
    calc = dxtb.Calculator(
        numbers,
        dxtb.GFN2_XTB,
        interaction=[spinpol],
        # "uhf_mode": True is optional because spinpol requires UHF.
        opts={"verbosity": 0},
        **dd,
    )
    result = calc.singlepoint(positions, chrg=0, spin=1)

``Calculator.nspin`` and ``Result.nspin`` report the resolved number of
wavefunction channels.  This avoids interpreting a dimension of length two as
a spin axis when a batch, atom, shell, orbital, or multipole dimension can
also have length two.

Charge and magnetization convention
-----------------------------------

Wavefunction tensors are stored as physical alpha/beta channels.  Population
derived tensors are stored as total-charge/magnetization channels, with the
total channel first and the magnetization channel second.  For Mulliken
electron populations :math:`p_\alpha` and :math:`p_\beta` and neutral
reference occupation :math:`n_0`, *dxtb* uses the tblite sign convention

.. math::

    q_\mathrm{total} &= n_0-p_\alpha-p_\beta, \\
    q_\mathrm{magnetization} &= -p_\alpha+p_\beta.

Consequently,

.. math::

    \sum q_\mathrm{total} = Q, \qquad
    \sum q_\mathrm{magnetization}
      = -(N_\alpha-N_\beta) = -\mathtt{spin}.

The same channel order and electron-charge sign apply to atomic dipoles and
quadrupoles.  Bond orders use total/magnetization order as well.  Internally,
the total and magnetization potentials produce the physical Hamiltonians

.. math::

    H_\alpha=H_\mathrm{total}+H_\mathrm{magnetization}, \qquad
    H_\beta=H_\mathrm{total}-H_\mathrm{magnetization}.

UHF result contract
-------------------

In the table below, ``...`` denotes any batch dimensions, ``nat`` the number
of atoms, and ``nao`` the number of atomic orbitals.  The spin axis is always
immediately before the physical vector or matrix dimensions.

.. list-table:: Shapes for a result with ``nspin == 2``
   :header-rows: 1

   * - Result quantity
     - Representation
     - Shape
   * - ``density``, ``hamiltonian``
     - alpha/beta
     - ``(..., 2, nao, nao)``
   * - ``coefficients``
     - alpha/beta
     - ``(..., 2, nao, nao)``
   * - ``emo``, ``occupation``
     - alpha/beta
     - ``(..., 2, nao)``
   * - ``charges.mono``, ``potential.mono``
     - total/magnetization
     - ``(..., 2, nao)``
   * - ``charges.dipole``, ``potential.dipole`` when present
     - total/magnetization
     - ``(..., 2, nat, 3)``
   * - ``charges.quad``, ``potential.quad`` when present
     - total/magnetization
     - ``(..., 2, nat, 6)``
   * - ``get_bond_orders()``
     - total/magnetization
     - ``(..., 2, nat, nat)``
   * - ``total``, ``scf``, ``fenergy``, ``cenergies`` values
     - atom-resolved total energy
     - ``(..., nat)``
   * - overlap and core-Hamiltonian integrals
     - spin-independent
     - ``(..., nao, nao)``
   * - dipole and quadrupole integrals when present
     - spin-independent
     - ``(..., 3, nao, nao)`` and ``(..., 6, nao, nao)``
   * - ``iter``
     - SCF iteration count
     - Python integer

Molecular scalar and vector observables, such as total energy, forces, and
electric moments, use the total channel unless their documentation explicitly
says otherwise.  GFN1-xTB normally has only monopoles; GFN2-xTB results can
also contain the dipole and quadrupole container fields.

Restricted calculations remain shape-compatible with earlier *dxtb*
versions: they do not gain a singleton spin dimension.  In particular,
matrices have shape ``(..., nao, nao)``, orbital energies and monopoles have
shape ``(..., nao)``, multipoles retain their atomic shapes, and bond orders
have shape ``(..., nat, nat)``.  The existing restricted ``occupation`` output
continues to have shape ``(..., 2, nao)`` because it records alpha/beta filling
of the shared spatial orbitals; ``nspin`` is still one.

Constants, scaling, dtype, and device
-------------------------------------

The standard spin constants cover physical elements H through Rn
(:math:`1\leq Z\leq86`).  Atomic number zero is reserved for padded batches
and has zero constants.  Elements above 86 are rejected.

``wscale`` is a scalar multiplier for all spin constants.  It defaults to
``1.0``; ``0.5`` halves the interaction energy and potential at fixed
magnetization, and ``0.0`` leaves the interaction present (and therefore still
selects UHF) while making its contribution zero.  A scalar floating-point
tensor can be supplied when derivatives with respect to the scale are needed.

Pass the same ``dtype`` and ``device`` to the calculator and
:func:`~dxtb.components.spin.new_spin_polarization`.  The factory materializes
the constants and scale with those settings, while ``numbers`` remains an
integer tensor on the same device.  A tensor scale remains in PyTorch's
autograd graph.

Command line
------------

For an open-shell coordinate file, the corresponding modes are:

.. code-block:: console

    $ dxtb molecule.xyz --spin 1
    $ dxtb molecule.xyz --spin 1 --unrestricted
    $ dxtb molecule.xyz --spin 1 --spin-polarized
    $ dxtb molecule.xyz --spin 1 --spin-polarized \
        --spin-polarization-scale 0.5

The scale option only takes effect together with ``--spin-polarized``.

For backward compatibility, the legacy spelling ``--uhf N`` remains an
integer alias for ``--spin N``: it specifies the number of unpaired electrons.
Changing it to a boolean would silently change the meaning of existing command
lines, so new scripts should use ``--unrestricted`` to select two wavefunction
channels.
