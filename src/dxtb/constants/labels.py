"""
Labels for methods.
"""

# xtb
GFN0_XTB = 0
GFN0_XTB_STRS = ("gfn0", "gfn0-xtb", "gfn0_xtb", "gfn0xtb")
GFN1_XTB = 1
GFN1_XTB_STRS = ("gfn1", "gfn1-xtb", "gfn1_xtb", "gfn1xtb")
GFN2_XTB = 2
GFN2_XTB_STRS = ("gfn2", "gfn2-xtb", "gfn2_xtb", "gfn2xtb")
GFN_XTB_MAP = ["GFN0-xTB", "GFN1-xTB", "GFN2-xTB"]

# guess
GUESS_EEQ = 0
GUESS_EEQ_STRS = ("eeq", "equilibration")
GUESS_SAD = 1
GUESS_SAD_STRS = ("sad", "zero")
GUESS_MAP = ["EEQ", "SAD"]

# integral driver
INTDRIVER_LIBCINT = 0
INTDRIVER_LIBCINT_STRS = ("libcint", "c")
INTDRIVER_AUTOGRAD = 1
INTDRIVER_AUTOGRAD_STRS = ("autograd", "pytorch", "torch", "dxtb")
INTDRIVER_ANALYTICAL = 2
INTDRIVER_ANALYTICAL_STRS = ("analytical", "pytorch2", "torch2", "dxtb2")
INTDRIVER_LEGACY = 3
INTDRIVER_LEGACY_STRS = ("legacy", "old", "loop")
INTDRIVER_MAP = ["libcint", "Autograd", "Analytical", "Legacy (loops)"]

# SCF
SCF_MODE_FULL = 0
SCF_MODE_FULL_STRS = ("full", "full_tracking", "unrolling")
SCF_MODE_IMPLICIT = 1
SCF_MODE_IMPLICIT_STRS = ("default", "implicit")
SCF_MODE_EXPERIMENTAL = 2
SCF_MODE_EXPERIMENTAL_STRS = ("experimental", "perfect")
SCF_MODE_MAP = ["Full Tracking (unrolling)", "implicit", "experimental"]

SCP_MODE_FOCK = 0
SCP_MODE_FOCK_STRS = ("fock", "fockian")
SCP_MODE_CHARGE = 1
SCP_MODE_CHARGE_STRS = ("charge", "charges")
SCP_MODE_POTENTIAL = 2
SCP_MODE_POTENTIAL_STRS = ("potential", "pot")
SCP_MODE_MAP = ["Fock matrix", "charges", "potential"]

FERMI_PARTITION_EQUAL = 0
FERMI_PARTITION_EQUAL_STRS = ("equal", "same")
FERMI_PARTITION_ATOMIC = 1
FERMI_PARTITION_ATOMIC_STRS = ("atom", "atomic")
FERMI_PARTITION_MAP = ["equal", "atom"]
