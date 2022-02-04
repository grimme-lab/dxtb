
import numpy as np

""" Definition of basic data classes """


# Maximum contraction length of basis functions.
# The limit is chosen as twice the maximum size returned by the STO-NG expansion
maxg = 12

class Cgto_Type():
    """  Contracted Gaussian type basis function """

    def __init__(self):
        # Angular momentum of this basis function
        self.ang = -1
        # Contraction length of this basis function
        self.nprim = 0
        # Exponent of the primitive Gaussian functions
        self.alpha = np.array([0.0 for _ in range(maxg)])
        # Contraction coefficients of the primitive Gaussian functions,
        # might contain normalization
        self.coeff = np.array([0.0 for _ in range(maxg)])

    def __str__(self):
        return f"cgto( l:{self.ang} | ng:{self.nprim} | alpha:{self.alpha} | coeff:{self.coeff} )"

