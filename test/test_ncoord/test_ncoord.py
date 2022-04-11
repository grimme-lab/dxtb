from unittest import TestCase
import torch

from xtbml import adjlist, data as atomic_data, utils
from xtbml.ncoord import ncoord
from xtbml.exlibs.tbmalt import Geometry


samples = {
    "PbH4-BiH3": dict(
        symbols="Pb H H H H Bi H H H".split(),
        positions=torch.Tensor(
            [
                [-0.00000020988889, -4.98043478877778, +0.00000000000000],
                [+3.06964045311111, -6.06324400177778, +0.00000000000000],
                [-1.53482054188889, -6.06324400177778, -2.65838526500000],
                [-1.53482054188889, -6.06324400177778, +2.65838526500000],
                [-0.00000020988889, -1.72196703577778, +0.00000000000000],
                [-0.00000020988889, +4.77334244722222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, -2.35039772300000],
                [-2.71400388988889, +6.70626379422222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, +2.35039772300000],
            ]
        ),
    ),
}

torch.set_printoptions(precision=10)


class TestCoordinationNumber(TestCase):
    """
    Test for coordination number
    """

    def test_cn_single(self):
        sample = samples["PbH4-BiH3"]
        geometry = Geometry(utils.symbol2number(sample["symbols"]), sample["positions"])
        rcov = atomic_data.covrad.covalent_rad_d3
        ref = torch.Tensor(
            [
                4.0012888908,
                0.9989894032,
                0.9989894032,
                0.9989894032,
                1.0055069923,
                3.0004417896,
                0.9962375164,
                0.9962376356,
                0.9962375164,
            ]
        )

        for symmetric in [True, False]:
            edges = adjlist.get_adjacency_map(geometry.positions, symmetric=symmetric)
            distances = adjlist.get_distances(geometry.positions, edges)
            cn = ncoord.get_coordination_number_d3(
                geometry.atomic_numbers,
                distances,
                edges,
                rcov,
                symmetric=symmetric,
            )
            self.assertTrue(torch.allclose(cn, ref))
