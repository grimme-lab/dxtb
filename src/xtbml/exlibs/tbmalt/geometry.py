# -*- coding: utf-8 -*-
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""A container to hold data associated with a chemical system's structure.

This module provides the `Geometry` data structure class and its associated
code. The `Geometry` class is intended to hold any & all data needed to fully
describe a chemical system's structure.
"""
from typing import Union, List, Optional
from operator import itemgetter
import torch
import numpy as np
from h5py import Group
from ase import Atoms

from .batch import pack, merge
from .units import length_units
from .data import chemical_symbols

Tensor = torch.Tensor


class Geometry:
    """Data structure for storing geometric information on molecular systems.

    The `Geometry` class stores any information that is needed to describe a
    chemical system; atomic numbers, positions, etc. This class also permits
    batch system representation. However, mixing of PBC & non-PBC systems is
    strictly forbidden.

    Arguments:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        charges : Charges of ionic molecules.
        units: Unit in which ``positions`` were specified. For a list of
            available units see :mod:`.units`. [DEFAULT='bohr']

    Attributes:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        n_atoms: Number of atoms in the system.

    Notes:
        When representing multiple systems, the `atomic_numbers` & `positions`
        tensors will be padded with zeros. Tensors generated from ase atoms
        objects or HDF5 database entities will not share memory with their
        associated numpy arrays, nor will they inherit their dtype.

    Warnings:
        At this time, periodic boundary conditions are not supported.

    Examples:
        Geometry instances may be created by directly passing in the atomic
        numbers & atom positions

        >>> from tbmalt import Geometry
        >>> H2 = Geometry(torch.tensor([1, 1]),
        >>>               torch.tensor([[0.00, 0.00, 0.00],
        >>>                             [0.00, 0.00, 0.79]]))
        >>> print(H2)
        Geometry(H2)

        Or from an ase.Atoms object

        >>> from ase.build import molecule
        >>> CH4_atoms = molecule('CH4')
        >>> print(CH4_atoms)
        Atoms(symbols='CH4', pbc=False)
        >>> CH4 = Geometry.from_ase_atoms(CH4_atoms)
        >>> print(CH4)
        Geometry(CH4)

        Multiple systems can be represented by a single ``Geometry`` instance.
        To do this, simply pass in lists or packed tensors where appropriate.

    """

    __slots__ = [
        "atomic_numbers",
        "positions",
        "n_atoms",
        "_n_batch",
        "_mask_dist",
        "__dtype",
        "__device",
        "_charges",
        "_unpaired_e",
    ]

    def __init__(
        self,
        atomic_numbers: Union[Tensor, List[Tensor]],
        positions: Union[Tensor, List[Tensor]],
        charges: Optional[Tensor] = None,
        unpaired_e: Optional[Tensor] = None,
        units: Optional[str] = "bohr",
    ):

        # "pack" will only effect lists of tensors
        self.atomic_numbers = pack(atomic_numbers)
        self.positions: Tensor = pack(positions)

        # Mask for clearing padding values in the distance matrix.
        if (temp_mask := self.atomic_numbers != 0).all():
            self._mask_dist: Union[Tensor, bool] = False
        else:
            self._mask_dist: Union[Tensor, bool] = ~(
                temp_mask.unsqueeze(-2) * temp_mask.unsqueeze(-1)
            )

        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (
            None if self.atomic_numbers.dim() == 1 else len(atomic_numbers)
        )

        # Ensure the distances are in atomic units (bohr)
        if units != "bohr":
            self.positions: Tensor = self.positions * length_units[units]

        # These are static, private variables and must NEVER be modified!
        self.__device = self.positions.device
        self.__dtype = self.positions.dtype

        # Check for size discrepancies in `positions` & `atomic_numbers`
        if self.atomic_numbers.ndim == 2:
            check = len(atomic_numbers) == len(positions)
            assert check, "`atomic_numbers` & `positions` size mismatch found"

        # Ensure tensors are on the same device (only two present currently)
        if self.positions.device != self.positions.device:
            raise RuntimeError("All tensors must be on the same device!")

        self._charges = (
            charges if charges != None else torch.zeros(self.positions.shape[0])
        )
        self._unpaired_e = (
            unpaired_e if unpaired_e != None else torch.zeros(self.positions.shape[0])
        )

    @property
    def device(self) -> torch.device:
        """The device on which the `Geometry` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        raise AttributeError(
            "Geometry object's dtype can only be modified " 'via the ".to" method.'
        )

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""
        dist = torch.cdist(self.positions, self.positions, p=2)
        # Ensure padding area is zeroed out
        dist[self._mask_dist] = 0
        return dist

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
        dist_vec[self._mask_dist] = 0
        return dist_vec

    @property
    def chemical_symbols(self) -> list:
        """Chemical symbols of the atoms present."""
        return batch_chemical_symbols(self.atomic_numbers)

    @property
    def batchsize(self) -> int:
        return self.atomic_numbers[0]

    @property
    def charges(self) -> Tensor:
        """Charges of ionic molecules."""
        return self._charges

    @property
    def unpaired_e(self) -> Tensor:
        """Number of unpaired electrons."""
        return self._unpaired_e

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return torch.unique(self.atomic_numbers[self.atomic_numbers.ne(0)])

        # a = self.atomic_numbers[self.atomic_numbers.ne(0)]
        # indexes = np.unique(a, return_index=True)[1]
        # return torch.tensor([a[index] for index in sorted(indexes)])

    def unique_chemical_symbols(self) -> List:
        """Identifies and returns a tensor of unique chemical symbols.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_chemical_symbols: A tensor specifying the unique chemical symbols present.
        """
        return batch_chemical_symbols(self.unique_atomic_numbers())

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: Union[Atoms, List[Atoms]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        units: str = "angstrom",
    ) -> "Geometry":
        """Instantiates a Geometry instance from an `ase.Atoms` object.

        Multiple atoms objects can be passed in to generate a batched Geometry
        instance which represents multiple systems.

        Arguments:
            atoms: Atoms object(s) to instantiate a Geometry instance from.
            device: Device on which to create any new tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Length unit used by `Atoms` object. [DEFAULT='angstrom']

        Returns:
            geometry: The resulting ``Geometry`` object.

        Warnings:
            Periodic boundary conditions are not currently supported.

        Raises:
            NotImplementedError: If the `ase.Atoms` object has periodic
                boundary conditions enabled along any axis.
        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        # Temporary catch for periodic systems
        def check_not_periodic(atom_instance):
            """Just raises an error if a system is periodic."""
            if atom_instance.pbc.any():
                raise NotImplementedError("TBMaLT does not support PBC")

        if not isinstance(atoms, list):  # If a single system
            check_not_periodic(atoms)  # Ensure non PBC system
            return cls(  # Create a Geometry instance and return it
                torch.tensor(atoms.get_atomic_numbers(), device=device),
                torch.tensor(atoms.positions, device=device, dtype=dtype),
                units=units,
            )

        else:  # If a batch of systems
            for a in atoms:  # Ensure non PBC systems
                check_not_periodic(a)
            return cls(  # Create a batched Geometry instance and return it
                [torch.tensor(a.get_atomic_numbers(), device=device) for a in atoms],
                [torch.tensor(a.positions, device=device, dtype=dtype) for a in atoms],
                units=units,
            )

    @classmethod
    def from_hdf5(
        cls,
        source: Union[Group, List[Group]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        units: str = "bohr",
    ) -> "Geometry":
        """Instantiate a `Geometry` instances from an HDF5 group.

        Construct a `Geometry` entity using data from an HDF5 group. Passing
        multiple groups, or a single group representing multiple systems, will
        return a batched `Geometry` instance.

        Arguments:
            source: An HDF5 group(s) containing geometry data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Unit of length used by the data. [DEFAULT='bohr']

        Returns:
            geometry: The resulting ``Geometry`` object.

        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        # If a single system or a batch system
        if not isinstance(source, list):
            # Read & parse a datasets from the database into a System instance
            # & return the result.
            return cls(
                torch.tensor(source["atomic_numbers"], device=device),
                torch.tensor(source["positions"], dtype=dtype, device=device),
                units=units,
            )
        else:
            return cls(  # Create a batched Geometry instance and return it
                [torch.tensor(s["atomic_numbers"], device=device) for s in source],
                [
                    torch.tensor(s["positions"], device=device, dtype=dtype)
                    for s in source
                ],
                units=units,
            )

    def to_hdf5(self, target: Group):
        """Saves Geometry instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the system's data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for atomic_numbers, positions, lattice, and pbc
        add_data("atomic_numbers", data=self.atomic_numbers.cpu().numpy())
        pos = add_data("positions", data=self.positions.cpu().numpy())

        # Add units meta-data to the atomic positions
        pos.attrs["unit"] = "bohr"

    def to(self, device: torch.device) -> "Geometry":
        """Returns a copy of the `Geometry` instance on the specified device

        This method creates and returns a new copy of the `Geometry` instance
        on the specified device "``device``".

        Arguments:
            device: Device to which all associated tensors should be moved.

        Returns:
            geometry: A copy of the `Geometry` instance placed on the
                specified device.

        Notes:
            If the `Geometry` instance is already on the desired device then
            `self` will be returned.
        """
        # Developers Notes: It is imperative that this function gets updated
        # whenever new attributes are added to the `Geometry` class. Otherwise
        # this will return an incomplete `Geometry` object.
        if self.atomic_numbers.device == device:
            return self
        else:
            return self.__class__(
                self.atomic_numbers.to(device=device), self.positions.to(device=device)
            )

    def __len__(self) -> int:
        """Returns length of geometry batch."""
        if len(self.positions.shape) == 2:
            return 1
        else:
            return self.positions.shape[0]

    def __getitem__(self, selector) -> "Geometry":
        """Permits batched Geometry instances to be sliced as needed."""
        # Block this if the instance has only a single system
        if self.atomic_numbers.ndim != 2:
            raise IndexError(
                "Geometry slicing is only applicable to batches of systems."
            )

        return self.__class__(
            self.atomic_numbers[selector],
            self.positions[selector],
            self.charges[selector],
            self.unpaired_e[selector],
        )

    def __add__(self, other: "Geometry") -> "Geometry":
        """Combine two `Geometry` objects together."""
        if self.__class__ != other.__class__:
            raise TypeError(
                "Addition can only take place between two Geometry objects."
            )

        # Catch for situations where one or both systems are not batched.
        s_batch = self.atomic_numbers.ndim == 2
        o_batch = other.atomic_numbers.ndim == 2

        an_1 = torch.atleast_2d(self.atomic_numbers)
        an_2 = torch.atleast_2d(other.atomic_numbers)

        pos_1 = self.positions
        pos_2 = other.positions

        pos_1 = pos_1 if s_batch else pos_1.unsqueeze(0)
        pos_2 = pos_2 if o_batch else pos_2.unsqueeze(0)

        chrg_1 = self.charges
        chrg_2 = other.charges
        try:
            chrg = torch.stack((chrg_1, chrg_2))
        except RuntimeError:
            chrg = torch.concat((chrg_1, chrg_2.unsqueeze(0)), dim=0)
        ue_1 = self.unpaired_e
        ue_2 = other.unpaired_e
        try:
            ue = torch.stack((ue_1, ue_2))
        except RuntimeError:
            ue = torch.concat((ue_1, ue_2.unsqueeze(0)), dim=0)

        return self.__class__(merge([an_1, an_2]), merge([pos_1, pos_2]), chrg, ue)

    def __eq__(self, other: "Geometry") -> bool:
        """Check if two `Geometry` objects are equivalent."""
        # Note that batches with identical systems but a different order will
        # return False, not True.

        if self.__class__ != other.__class__:
            raise TypeError(
                f'"{self.__class__}" ==  "{other.__class__}" '
                f"evaluation not implemented."
            )

        def shape_and_value(a, b):
            return a.shape == b.shape and torch.allclose(a, b)

        return all(
            [
                shape_and_value(self.atomic_numbers, other.atomic_numbers),
                shape_and_value(self.positions, other.positions),
                shape_and_value(self.charges, other.charges),
                shape_and_value(self.unpaired_e, other.unpaired_e),
            ]
        )

    def __repr__(self) -> str:
        """Creates a string representation of the Geometry object."""
        # Return Geometry(CH4) for a single system & Geometry(CH4, H2O, ...)
        # for multiple systems. Only the first & last two systems get shown if
        # there are more than four systems (this prevents endless spam).

        def get_formula(atomic_numbers: Tensor) -> str:
            """Helper function to get reduced formula."""
            # If n atoms > 30; then use the reduced formula
            if len(atomic_numbers) > 30:
                return "".join(
                    [
                        f"{chemical_symbols[z]}{n}"
                        if n != 1
                        else f"{chemical_symbols[z]}"
                        for z, n in zip(*atomic_numbers.unique(return_counts=True))
                        if z != 0
                    ]
                )  # <- Ignore zeros (padding)

            # Otherwise list the elements in the order they were specified
            else:
                return "".join(
                    [
                        f"{chemical_symbols[int(z)]}{int(n)}"
                        if n != 1
                        else f"{chemical_symbols[z]}"
                        for z, n in zip(
                            *torch.unique_consecutive(
                                atomic_numbers, return_counts=True
                            )
                        )
                        if z != 0
                    ]
                )

        if self.atomic_numbers.dim() == 1:  # If a single system
            formula = get_formula(self.atomic_numbers)
        else:  # If multiple systems
            if self.atomic_numbers.shape[0] <= 4:  # If n<4 systems; show all
                formulas = [get_formula(an) for an in self.atomic_numbers]
                formula = " ,".join(formulas)
            else:  # If n>4; show only the first and last two systems
                formulas = [
                    get_formula(an) for an in self.atomic_numbers[[0, 1, -2, -1]]
                ]
                formula = "{}, {}, ..., {}, {}".format(*formulas)

        # Wrap the formula(s) in the class name and return
        return f"{self.__class__.__name__}({formula})"

    def __str__(self) -> str:
        """Creates a printable representation of the System."""
        # Just redirect to the `__repr__` method
        return repr(self)

    def get_length(self, unique: bool = False) -> int:
        """
        Return number of (unique) atoms in geometry over whole batch.
        """
        if unique:
            # number of unique elements
            l = len(self.unique_atomic_numbers())
        else:
            # total number of elements
            l = torch.numel(self.atomic_numbers)
        return l

    def get_species_index(self, atomic_number: Union[int, Tensor]) -> Union[int, None]:
        """
        Return index of species in object specific data tensor.

        Args:
                atomic_number (int): Atomic number of species. Single value tensor allowed.

        Returns:
                int: Position inside internal data model
        """
        try:
            return (
                (self.unique_atomic_numbers() == atomic_number)
                .nonzero(as_tuple=True)[0]
                .item()
            )
        except ValueError:
            if isinstance(atomic_number, Tensor):
                if atomic_number.item() == 0:
                    return None
            elif isinstance(atomic_number, int):
                if atomic_number == 0:
                    return None
            raise ValueError("No corresponding idx found.")

    def generate_interactions(self, unique: bool = False) -> Tensor:
        """
        Identify all unique species combinations.

        Args:
            self (Geometry): System geometry either as batch-pair or single

        Yields:
            Iterator[Tensor]: Interaction of species combination (pair of atomic numbers)
        """

        # NOTE: this method can be extended and adapted for future geometry setups
        if unique:
            # Identify all unique species combinations.
            interactions = torch.combinations(
                self.unique_atomic_numbers(), with_replacement=True
            )
        else:
            # Permutation of all atom index pairs
            interactions = torch.combinations(
                torch.arange(0, self.atomic_numbers.shape[-1], dtype=torch.int),
                with_replacement=True,
            )

        # Loop over the unique interaction pairs
        for interaction in interactions:
            yield interaction


####################
# Helper Functions #
####################
def batch_chemical_symbols(atomic_numbers: Union[Tensor, List[Tensor]]) -> list:
    """Converts atomic numbers to their chemical symbols.

    This function allows for en-mass conversion of atomic numbers to chemical
    symbols.

    Arguments:
        atomic_numbers: Atomic numbers of the elements.

    Returns:
        symbols: The corresponding chemical symbols.

    Notes:
        Padding vales, i.e. zeros, will be ignored.

    """
    a_nums = atomic_numbers

    # Catch for list tensors (still faster doing it this way)
    if isinstance(a_nums, list) and isinstance(a_nums[0], Tensor):
        a_nums = pack(a_nums, value=0)

    # Convert from atomic numbers to chemical symbols via a itemgetter
    symbols = np.array(  # numpy must be used as torch cant handle strings
        itemgetter(*a_nums.flatten())(chemical_symbols)
    ).reshape(a_nums.shape)
    # Mask out element "X", aka padding values
    mask = symbols != "X"
    if symbols.ndim == 1:
        return symbols[mask].tolist()
    else:
        return [s[m].tolist() for s, m in zip(symbols, mask)]


def unique_atom_pairs(geometry: Geometry) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """
    uan = geometry.unique_atomic_numbers()
    n_global = len(uan)
    return torch.stack([uan.repeat(n_global), uan.repeat_interleave(n_global)]).T
