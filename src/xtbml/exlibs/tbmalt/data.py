# -*- coding: utf-8 -*-
# NOTE: THANKS TO THE COURTESY OF https://github.com/tbmalt/tbmalt
"""Constants & reference data required by TBMaLT is stored in the data module.

The data module stores any static data needed by TBMaLT or its users. This
includes, but is not limited to, constants, conversion factors, reference
data, etc. Attributes present in the module level namespace (:mod:`data`) are
documented here. However, those stored in sub-domains, such as :mod:`.units`,
are documented in their respective module sections.


Attributes:
    chemical_symbols (List[str]): List of chemical symbols whose indices are
        the atomic numbers of the associated elements; i.e.
        `chemical_symbols[6]` will yield `"C"`.
    atomic_numbers (Dict[str, int]): Dictionary keyed by chemical symbols &
        valued by atomic numbers. This is used to get the atomic number
        associated with a given chemical symbol.

"""
# Make chemical_symbols/atomic_numbers accessible in the tbmalt.data namespace
from .elements import chemical_symbols, atomic_numbers
