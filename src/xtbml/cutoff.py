import math
import torch

# TODO define interface


def get_lattice_points(a, b, c) -> torch.Tensor:
    """Realspace cutoff and lattice point generator utilities."""
    try:
        periodic, lat, rthr = a, b, c
        trans = get_lattice_points_cutoff(periodic, lat, rthr)
    except (AttributeError, ValueError) as e:  # TODO: check error
        lat, rep, origin = a, b, c
        trans = get_lattice_points_rep_3d(lat, rep, origin)
    return trans


def get_lattice_points_rep_3d(lat, rep, origin):
    """Generate lattice points from repeatitions

    Args:
        lat (_type_): Lattice vectors
        rep (_type_): Repeatitions of lattice points to generate
        origin (_type_): Include the origin in the generated lattice points
        trans (_type_): Generated lattice points
    """

    """
    real(wp), intent(in) :: lat(:, :)
    # Repeatitions of lattice points to generate
    integer, intent(in) :: rep(:)
    # Include the origin in the generated lattice points
    logical, intent(in) :: origin
    # Generated lattice points
    real(wp), allocatable, intent(out) :: trans(:, :)"""

    itr = 0
    if origin is not None:
        # TODO: allocate(trans(3, product(2*rep+1)))
        trans = torch.zeros([3, torch.prod(2 * rep + 1)])
        for ix in range(rep[1] + 1):
            for iy in range(rep[2] + 1):
                for iz in range(rep[3] + 1):
                    # TODO: merge(-1, 1, ix > 0), -2
                    for jx in range(torch.where(ix > 0, -1, 1), -2):
                        for jy in range(torch.where(iy > 0, -1, 1), -2):
                            for jz in range(torch.where(iz > 0, -1, 1), -2):
                                trans[:, itr] = (
                                    lat[:, 0] * ix * jx
                                    + lat[:, 1] * iy * jy
                                    + lat[:, 2] * iz * jz
                                )
                                itr += 1
    else:
        # TODO: allocate(trans(3, product(2*rep+1)-1))
        trans = torch.zeros([3, torch.prod(2 * rep + 1) - 1])
        for ix in range(rep[1] + 1):
            for iy in range(rep[2] + 1):
                for iz in range(rep[3] + 1):
                    if all([ix == 0, iy == 0, iz == 0]):
                        continue
                    for jx in range(torch.where(ix > 0, -1, 1), -2):
                        for jy in range(torch.where(iy > 0, -1, 1), -2):
                            for jz in range(torch.where(iz > 0, -1, 1), -2):
                                trans[:, itr] = (
                                    lat[:, 0] * ix * jx
                                    + lat[:, 1] * iy * jy
                                    + lat[:, 2] * iz * jz
                                )
                                itr += 1
    return trans


def get_lattice_points_cutoff(periodic, lat, rthr):
    """Create lattice points within a given cutoff"""

    """# Periodic dimensions
    logical, intent(in) :: periodic(:)
    # Real space cutoff
    real(wp), intent(in) :: rthr
    # Lattice parameters
    real(wp), intent(in) :: lat(:, :)
    # Generated lattice points
    real(wp), allocatable, intent(out) :: trans(:, :)"""

    if not any(periodic):
        trans = torch.zeros([3, 1])
    else:
        rep = get_translations(lat, rthr)
        trans = get_lattice_points(lat, rep, True)

    return trans


def get_translations(lat, rthr):
    # Generate a supercell based on a realspace cutoff, this subroutine
    # doesn't know anything about the convergence behaviour of the
    # associated property.
    """real(wp), intent(in) :: rthr
    real(wp), intent(in) :: lat(3, 3)
    integer, intent(out) :: rep(3)
    real(wp) :: normx(3), normy(3), normz(3)
    real(wp) :: cos10, cos21, cos32"""

    rep = torch.zeros([3])

    def crossproduct(a, b):
        """real(wp), intent(in) :: a(3)
        real(wp), intent(in) :: b(3)
        real(wp), intent(out) :: c(3)"""

        c = torch.zeros([3])
        c[0] = a[1] * b[2] - b[1] * a[2]
        c[1] = a[2] * b[0] - b[2] * a[0]
        c[2] = a[0] * b[1] - b[0] * a[1]
        return c

    # find normal to the plane...
    normx = crossproduct(lat[:, 1], lat[:, 2])
    normy = crossproduct(lat[:, 2], lat[:, 0])
    normz = crossproduct(lat[:, 0], lat[:, 1])

    # ...normalize it...
    normx = normx / torch.linalg.norm(normx)
    normy = normy / torch.linalg.norm(normy)
    normz = normz / torch.linalg.norm(normz)
    # cos angles between normals and lattice vectors
    cos10 = torch.sum(normx * lat[:, 0])
    cos21 = torch.sum(normy * lat[:, 1])
    cos32 = torch.sum(normz * lat[:, 2])

    rep[0] = math.ceil(abs(rthr / cos10))
    rep[1] = math.ceil(abs(rthr / cos21))
    rep[2] = math.ceil(abs(rthr / cos32))

    return rep


def wrap_to_central_cell(xyz, lattice, periodic):
    """real(wp), intent(inout) :: xyz(:, :)
    real(wp), intent(in) :: lattice(:, :)
    logical, intent(in) :: periodic(:)
    real(wp) :: invlat(3, 3), vec(3)
    integer :: iat, idir"""

    if not any(periodic):
        return

    # use mctc_io_math, only : matinv_3x3
    # invlat = matinv_3x3(lattice)
    invlat = torch.linalg.inv(lattice)  # TODO

    for iat in range(xyz.shape[2]):
        vec = invlat @ xyz[:, iat]
        vec = shift_back_abc(vec)
        xyz[:, iat] = lattice @ vec
    return xyz


def shift_back_abc(inp: float):  # TODO
    """# fractional coordinate in (-∞,+∞)
    real(wp),intent(in) :: in
    # fractional coordinate in [0,1)
    real(wp) :: out"""

    p_pbc_eps = 1.0e-14
    out = inp
    if inp < (0.0 - p_pbc_eps):
        out = inp + float(math.ceil(-inp))
    if inp > (1.0 + p_pbc_eps):
        out = inp - float(math.floor(inp))
    if math.fabs(inp - 1.0) < p_pbc_eps:
        out = inp - 1.0
    return out
