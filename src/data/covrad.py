


from typing import Union

# TODO define interface
def get_covalent_rad(value: Union[str,int]) -> int:
    """ Covalent radii for DFT-D3 coordination number.
    """
    try:
        rad = get_covalent_rad_num(value)
    except (AttributeError, ValueError) as e: # TODO: check error
        rad = get_covalent_rad_sym(value)
    return rad

#use mctc_io_convert, only : aatoau
aatoau = 1.8897259886 # TODO: check value
max_elem = 118

periodic_table = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 
                11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 
                21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 
                31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 
                41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 
                51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 
                61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 
                71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 
                81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 
                91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 
                101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 
                110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og",}

# Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
#  188-197), values for metals decreased by 10 %
covalent_rad_2009 = aatoau * [ 
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
                     1.36,1.43,1.46,1.58,1.48,1.57 ] # Nh-Og

# D3 covalent radii used to construct the coordination number
covalent_rad_d3 = 4.0 / 3.0 * covalent_rad_2009


def to_number(sym: str):
    return [i for i, n in periodic_table.items() if n == sym]


def get_covalent_rad_sym(sym: str) -> float:
    """ Get covalent radius for a given element symbol

    Args:
        sym (str): Element symbol

    Returns:
        float: Covalent radius
    """
    return get_covalent_rad(*to_number(sym))


def  get_covalent_rad_num(num: int) -> float:
    """ Get covalent radius for a given atomic number

    Args:
        num (int): Atomic number

    Returns:
        float: Covalent radius
    """
    if num > 0 and num <= len(covalent_rad_d3):
        rad = covalent_rad_d3(num)
    else:
        rad = 0.0
    return rad
