.. _indepth_components_new:

Creating New Components
=======================

In the following, we will explain the process of creating a new tight-binding
component that can be added to the :class:`~dxtb.Calculator`. For a correct
evaluation within the :class:`~dxtb.Calculator`, the corresponding methods of
the base :class:`~dxtb.components.base.Component` class must be implemented. We
will show this step by step for the electric field (which itself is already
implemented).

Step 1: Create class.
~~~~~~~~~~~~~~~~~~~~~

Since the electric field interacts with the charges, the electric field
contributes to the charge-dependent Hamiltonian. For the implementation, this
means it is a "self-consistent" component, i.e., it should inherit from the
:class:`~dxtb.components.base.Interaction` class.

.. code-block:: python

   class ElectricField(Interaction):
       """
       Instantaneous electric field.
       """

Step 2: Add constructor and parameters.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While actual tight-binding components save the parametrization for each atom
in the attributes/fields, the electric field is fully described by the field
vector. Therefore, the constructor of the electric field only takes the field
vector as an argument and writes it to the ``field`` attribute.

.. code-block:: python

    class ElectricField(Interaction):
        """
        Instantaneous electric field.
        """

        field: Tensor
        """Instantaneous electric field vector."""

        __slots__ = ["field"]

    def __init__(
        self,
        field: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else field.device,
            dtype=dtype if dtype is None else field.dtype,
        )
        self.field = field

The constructor should always call the constructor of the base class with the
``dtype`` and ``device`` arguments to ensure consistency. Both arguments bubble
up to the :class:`~dxtb.components.base.Interaction` class, through the
:class:`~dxtb.components.base.Component` class, ending up in the
:class:`~dxtb.typing.TensorLike` class, which facilitates changing the
device and dtype of all tensors in the class with PyTorch's well known
:meth:`~torch.Tensor.to` and :meth:`~torch.Tensor.type` method. The
:class:`~dxtb.typing.TensorLike` class also registers the ``self.dtype``,
``self.device`` and ``self.dd`` properties.

Do not forget to add the ``__slots__`` attribute to the class. Otherwise, the
:meth:`~torch.Tensor.to` and :meth:`~torch.Tensor.type` methods will not work.
All ``__slots__`` should be arguments of the constructor.

Step 3: Create cache.
~~~~~~~~~~~~~~~~~~~~~

The internal :class:`Cache` should inherit from the cache of the base class
:class:`~dxtb.components.base.InteractionCache` (which makes it
:class:`~dxtb.typing.TensorLike` again). Correspondingly, the constructor
is similar to the one of the electric field itself. The attributes and
``__slots__`` are also initialized in the same way. For the electric field,
the cache contains the atom-resolved monopolar and dipolar potentials.

.. code-block:: python

    class ElectricFieldCache(InteractionCache):
        """
        Restart data for the electric field interaction.
        """

        vat: Tensor
        """
        Atom-resolved monopolar potental from instantaneous electric field.
        """

        vdp: Tensor
        """
        Atom-resolved dipolar potential from instantaneous electric field.
        """

        __slots__ = ["vat", "vdp"]

        def __init__(
            self,
            vat: Tensor,
            vdp: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(
                device=device if device is None else vat.device,
                dtype=dtype if dtype is None else vat.dtype,
            )
            self.vat = vat
            self.vdp = vdp

Step 4: Modify cache for culling in batched SCF.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is less straightforward. Essentially, the
:class:`~dxtb.components.base.Cache` must be updated if a system is removed
from the batch dimension upon convergence within the SCF ("culling").
Simultanously, all cache variables must be stored to allow restoring them after
the SCF for the final energy evaluation. Correspondingly, we add a simple
:class:`Store` class and a corresponding attribute (``__store``) to the
:class:`Cache`. The ``__store`` attribute is initialized to ``None`` and will
only be filled when the :meth:`cull` method is called. The :meth:`cull` method
takes the indices of systems that are removed from the batch (``conv`` tensor)
and a collection of ``slicers``, which are used for potentially resizing
tensors if the largest system was culled from the batch
(:class:`~dxtb.typing.Slicers` class). For the atom-resolved monopolar and
dipolar potentials, the corresponding atom-resolved slicers is collected. The
attributes are sliced, while a copy remains in the ``Store``. Restoring the
cache is done by the :meth:`restore` method, which simply copies the
:class:`Store` attributes back to the cache.

.. code-block:: python

    class ElectricFieldCache(InteractionCache, TensorLike):
      """
      Restart data for the electric field interaction.
      """

      __store: Store | None
      """Storage for cache (required for culling)."""

      vat: Tensor
      """
      Atom-resolved monopolar potental from instantaneous electric field.
      """

      vdp: Tensor
      """
      Atom-resolved dipolar potential from instantaneous electric field.
      """

      __slots__ = ["__store", "vat", "vdp"]

      def __init__(
          self,
          vat: Tensor,
          vdp: Tensor,
          device: torch.device | None = None,
          dtype: torch.dtype | None = None,
      ) -> None:
          super().__init__(
              device=device if device is None else vat.device,
              dtype=dtype if dtype is None else vat.dtype,
          )
          self.vat = vat
          self.vdp = vdp
          self.__store = None

      class Store:
          """
          Storage container for cache containing ``__slots__`` before culling.
          """

          vat: Tensor
          """
          Atom-resolved monopolar potental from instantaneous electric field.
          """

          vdp: Tensor
          """
          Atom-resolved dipolar potential from instantaneous electric field.
          """

          def __init__(self, vat: Tensor, vdp: Tensor) -> None:
              self.vat = vat
              self.vdp = vdp

      def cull(self, conv: Tensor, slicers: Slicers) -> None:
          if self.__store is None:
              self.__store = self.Store(self.vat, self.vdp)

          slicer = slicers["atom"]
          self.vat = self.vat[[~conv, *slicer]]
          self.vdp = self.vdp[[~conv, *slicer, ...]]

      def restore(self) -> None:
          if self.__store is None:
              raise RuntimeError("Nothing to restore. Store is empty.")

          self.vat = self.__store.vat
          self.vdp = self.__store.vdp

Step 5: Populate the cache (`get_cache`).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cachable quantities are computed within the :meth:`get_cache` method. The
:class:`Cache` is instantiated and returned. Note that if the interaction is
evaluated within the :class:`~dxtb.components.base.InteractionList`,
``numbers`` and :class:`~dxtb.IndexHelper` will be passed as argument, too.
This is done to fulfill the different requirements of the caches, while
retaining a (somewhat) consistent API. The electric field cache only needs the
position tensor. Correspondingly, the ``numbers`` and ``ihelp`` are not stored
in the ``cachvars`` tuple, which is used to check if the cache is up-to-date.

.. code-block:: python

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ElectricFieldCache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        ElectricFieldCache
            Restart data for the interaction.

        Note
        ----
        If this interaction is evaluated within the `InteractionList`, `numbers`
        and `IndexHelper` will be passed as argument, too. The `**_` in the
        argument list will absorb those unnecessary arguments which are given
        as keyword-only arguments (see `Interaction.get_cache()`).
        """
        if positions is None:
            raise ValueError("Electric field requires atomic positions.")

        cachvars = (positions.detach().clone(), self.field.detach().clone())

        if self.cache_is_latest(cachvars) is True:
            if not isinstance(self.cache, ElectricFieldCache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        self._cachevars = cachvars

        # (nbatch, natoms, 3) * (3) -> (nbatch, natoms)
        vat = einsum("...ik,k->...i", positions, self.field)

        # (nbatch, natoms, 3)
        vdp = self.field.expand_as(positions)

        self.cache = ElectricFieldCache(vat, vdp)
        return self.cache

Step 6: Implement the energy evaluation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The energy from the electric field has a monopolar and a dipolar contribution.
Hence, both a :meth:`get_atom_energy` and a :meth:`get_dipole_energy` method
must be implemented. They overwrite the corresponding methods of the base
class, which would evaluate to zero. In general, all methods that are not
implemented in the derived class will evaluate to zero.

.. code-block:: python

    @override
    def get_monopole_atom_energy(
        self, cache: ElectricFieldCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the monopolar contribution of the electric field energy.

        Parameters
        ----------
        cache : ElectricFieldCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """
        return -cache.vat * qat

    @override
    def get_dipole_atom_energy(
        self,
        cache: ElectricFieldCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the dipolar contribution of the electric field energy.

        Parameters
        ----------
        cache : ElectricFieldCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved shadow charges (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """
        assert qdp is not None

        # equivalent: torch.sum(-cache.vdp * qdp, dim=-1)
        return einsum("...ix,...ix->...i", -cache.vdp, qdp)

Step 7: Implement the potential evaluation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the energy evaluation, the potential evaluation is split into a
monopolar and a dipolar contribution (to the charge-dependent Hamiltonian).
For API consistency, the charges are passed as a dummy argument.

.. code-block:: python

    @override
    def get_monopole_atom_potential(
        self, cache: ElectricFieldCache, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the electric field potential.

        Parameters
        ----------
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field potential.
        """
        return -cache.vat

    @override
    def get_dipole_atom_potential(
        self, cache: ElectricFieldCache, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the electric field dipole potential.

        Parameters
        ----------
        cache : ElectricFieldCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field dipole potential.
        """
        return -cache.vdp

Step 8: String representation (optional).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As good practice, the :meth:`__str__` and :meth:`__repr__` methods should be
implemented to provide a human-readable representation of the component.

.. code-block:: python

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(field={self.field})"

    def __repr__(self) -> str:
        return str(self)

Step 9: Add to the Calculator.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the electric field in a calculation, it must be added to the
`Calculator`. This is done by passing an instance of the electric field to the
constructor of the `Calculator`.

.. code-block:: python

    import torch
    from dxtb.typing import DD
    from dxtb import Calculator, GFN1_XTB

    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

    field = torch.tensor([0.0, 0.0, 0.0], **dd)
    ef = ElectricField(field=field, **dd)

    numbers = torch.tensor([3, 1], **dd)
    calc = Calculator(
        numbers,
        GFN1_XTB,
        interactions=[ef]
    )
