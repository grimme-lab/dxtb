""" Gram-Schmidt orthonormalization routines for contracted Gaussian basis functions """

import math


def orthogonalize(cgtoi, cgtoj):
    """Orthogonalize a contracted Gaussian basis function to an existing basis function

    Args:
        cgtoi (Cgto_Type): Existing basis function
        cgtoj (Cgto_Type): Basis function to orthogonalize

    Returns:
        cgtoj (Cgto_Type): Orthogonalized basis function
    """

    if cgtoi.ang != cgtoj.ang:
        return cgtoj

    # cgtoi
    overlap = 0.0
    for ipr in range(cgtoi.nprim):
        for jpr in range(cgtoj.nprim):
            eab = cgtoi.alpha[ipr] + cgtoj.alpha[jpr]
            oab = 1.0 / eab
            kab = math.sqrt(math.pi * oab) ** 3
            overlap += cgtoi.coeff[ipr] * cgtoj.coeff[jpr] * kab

    cgtoj.alpha[cgtoj.nprim : cgtoj.nprim + cgtoi.nprim] = cgtoi.alpha[: cgtoi.nprim]
    cgtoj.coeff[cgtoj.nprim : cgtoj.nprim + cgtoi.nprim] = (
        -overlap * cgtoi.coeff[: cgtoi.nprim]
    )
    cgtoj.nprim += cgtoi.nprim

    # cgtoj
    overlap = 0.0
    for ipr in range(cgtoj.nprim):
        for jpr in range(cgtoj.nprim):
            eab = cgtoj.alpha[ipr] + cgtoj.alpha[jpr]
            oab = 1.0 / eab
            kab = math.sqrt(math.pi * oab) ** 3
            overlap = overlap + cgtoj.coeff[ipr] * cgtoj.coeff[jpr] * kab

    cgtoj.coeff[: cgtoj.nprim] = cgtoj.coeff[: cgtoj.nprim] / math.sqrt(overlap)

    return cgtoj
