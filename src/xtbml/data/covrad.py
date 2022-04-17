from typing import Dict, List, Union
from torch import tensor

from xtbml.constants import AA2AU, PSE

# TODO define interface


# fmt: off
# Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
#  188-197), values for metals decreased by 10 %
covalent_rad_2009 = AA2AU * tensor([ 
    0.00,  # None
    0.32,0.46,  # H,He
    1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67,  # Li-Ne
    1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96,  # Na-Ar
    1.76,1.54,  # K,Ca
    1.33,1.22,1.21,1.10,1.07,  # Sc-
    1.04,1.00,0.99,1.01,1.09,  # -Zn
    1.12,1.09,1.15,1.10,1.14,1.17,  # Ga-Kr
    1.89,1.67,  # Rb,Sr
    1.47,1.39,1.32,1.24,1.15,  # Y-
    1.13,1.13,1.08,1.15,1.23,  # -Cd
    1.28,1.26,1.26,1.23,1.32,1.31,  # In-Xe
    2.09,1.76,  # Cs,Ba
    1.62,1.47,1.58,1.57,1.56,1.55,1.51,  # La-Eu
    1.52,1.51,1.50,1.49,1.49,1.48,1.53,  # Gd-Yb
    1.46,1.37,1.31,1.23,1.18,  # Lu-
    1.16,1.11,1.12,1.13,1.32,  # -Hg
    1.30,1.30,1.36,1.31,1.38,1.42,  # Tl-Rn
    2.01,1.81,  # Fr,Ra
    1.67,1.58,1.52,1.53,1.54,1.55,1.49,  # Ac-Am
    1.49,1.51,1.51,1.48,1.50,1.56,1.58,  # Cm-No
    1.45,1.41,1.34,1.29,1.27,  # Lr-
    1.21,1.16,1.15,1.09,1.22,  # -Cn
    1.36,1.43,1.46,1.58,1.48,1.57 ]) # Nh-Og
# fmt: on

# D3 covalent radii used to construct the coordination number
covalent_rad_d3 = 4.0 / 3.0 * covalent_rad_2009


def get_covalent_rad(
    value: Union[str, int, List], dct: Union[None, Dict[str, float]] = None
) -> Dict[str, float]:
    """Covalent radii for DFT-D3 coordination number."""

    if dct is None:
        dct = {}

    if type(value) is str:
        dct[value] = get_covalent_rad_sym(value)
    elif type(value) is int:
        dct[to_string(value)] = get_covalent_rad_num(value)
    elif type(value) is list:
        for el in value:
            dct = get_covalent_rad(el, dct)
    else:
        raise TypeError

    return dct


def to_number(sym: str) -> int:
    for i, n in PSE.items():
        if n == sym:
            return i


def to_string(num: int) -> str:
    return PSE[num]


def get_covalent_rad_sym(sym: str) -> float:
    """Get covalent radius for a given element symbol

    Args:
        sym (str): Element symbol

    Returns:
        float: Covalent radius
    """
    return get_covalent_rad_num(to_number(sym))


def get_covalent_rad_num(num: int) -> float:
    """Get covalent radius for a given atomic number

    Args:
        num (int): Atomic number

    Returns:
        float: Covalent radius
    """
    if num > 0 and num <= len(covalent_rad_d3):
        rad = covalent_rad_d3[num].item()
    else:
        rad = 0.0
    return rad
