.. _quickstart-cli:

Command Line Interface
======================

The CLI keeps the molecular spin state, unrestricted wavefunction mode, and
spin-polarization interaction separate.  For example:

.. code-block:: console

    $ dxtb molecule.xyz --spin 1
    $ dxtb molecule.xyz --spin 1 --unrestricted
    $ dxtb molecule.xyz --spin 1 --spin-polarized

Here, ``--spin 1`` requests one unpaired electron, ``--unrestricted`` is a
boolean two-channel mode switch, and ``--spin-polarized`` adds the
self-consistent energy interaction (which also implies unrestricted mode).
The option ``--uhf 1`` remains an integer alias for ``--spin 1``; it is not
the boolean mode flag.  See :ref:`quickstart-spin` for scaling, result
conventions, and complete examples.

.. automodule:: dxtb._src.cli
