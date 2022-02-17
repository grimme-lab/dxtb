
from unittest import TestCase
import torch

from basis.type import Cgto_Type
from basis.slater import slater_to_gauss
from integral.overlap import overlap_cgto

""" Testing the functionality of the overlap. """

# equality threshold
thr_atol = 1e-05
class Test_Slater_Expansion(TestCase):

    @classmethod
    def setUpClass(cls):
        print("Test_Slater_Expansion")
        
    def norm_l(self, l: int, dim: int, n: int, ng: int):
        """ Method for testing orbitals with momentum quantum number l

        Args:
            l (int): Momentum quantum number
            dim (int): Dimension of self-overlap matrix
            n (int): Main quantum number
            ng (int): Number of gaussians
        """

        print(f"test_l{l}_{dim}_n{n}_ng{ng}")
        import os.path as op
        proj_dir = op.join(op.dirname(op.abspath(__file__)), "..")
    
        with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(proj_dir), with_stack=True,  profile_memory=True) as prof:
            # same site 
            r2 = 0.0
            vec = torch.tensor([0., 0., 0.])

            with torch.profiler.record_function("slater_to_gauss"):
                # create gaussians
                cgto = Cgto_Type()
                slater_to_gauss(ng, n, l, 1.0, cgto, True)

            #with torch.profiler.record_function("overlap_cgto"):
            if True:
                # calculate self-overlap
                overlap = overlap_cgto(cgto, cgto, r2, vec, 100.0)

            with torch.profiler.record_function("assertTrue"):
                # self-overlap should be identity matrix
                target = torch.eye(dim)

                self.assertTrue(torch.allclose(overlap, target, rtol=1e-05, atol=thr_atol, equal_nan=False), msg=f"Self overlap not identity:\n {overlap}")

        print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))
        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
        
        return

    # assert self-overlap for different orbitals
    def assert_norm_s(self, n: int, ng: int):
        """ Wrapper method for testing s-orbitals. """
        return self.norm_l(0, 1, n, ng)

    def assert_norm_p(self, n: int, ng: int):
        """ Wrapper method for testing p-orbitals. """
        return self.norm_l(1, 3, n, ng)

    def assert_norm_d(self, n: int, ng: int):
        """ Wrapper method for testing d-orbitals. """
        return self.norm_l(2, 5, n, ng)

    def assert_norm_f(self, n: int, ng: int):
        """ Wrapper method for testing f-orbitals. """
        return self.norm_l(3, 7, n, ng)

    def assert_norm_g(self, n: int, ng: int):
        """ Wrapper method for testing g-orbitals. """
        return self.norm_l(4, 9, n, ng)


    # s-orbitals
    def test_norm_1s_sto1g(self):
        self.assert_norm_s(1, 1)

    """def test_norm_2s_sto1g(self):
        self.assert_norm_s(2, 1)
    
    def test_norm_3s_sto1g(self):
        self.assert_norm_s(3, 1)
    
    def test_norm_4s_sto1g(self):
        self.assert_norm_s(4, 1)
    
    def test_norm_5s_sto1g(self):
        self.assert_norm_s(5, 1)
    

    def test_norm_1s_sto2g(self):
        self.assert_norm_s(1, 2)
    
    def test_norm_2s_sto2g(self):
        self.assert_norm_s(2, 2)
    
    def test_norm_3s_sto2g(self):
        self.assert_norm_s(3, 2)
    
    def test_norm_4s_sto2g(self):
        self.assert_norm_s(4, 2)
    
    def test_norm_5s_sto2g(self):
        self.assert_norm_s(5, 2)
    

    def test_norm_1s_sto3g(self):
        self.assert_norm_s(1, 3)
    
    def test_norm_2s_sto3g(self):
        self.assert_norm_s(2, 3)
    
    def test_norm_3s_sto3g(self):
        self.assert_norm_s(3, 3)
    
    def test_norm_4s_sto3g(self):
        self.assert_norm_s(4, 3)
    
    def test_norm_5s_sto3g(self):
        self.assert_norm_s(5, 3)
    

    def test_norm_1s_sto4g(self):
        self.assert_norm_s(1, 4)
    
    def test_norm_2s_sto4g(self):
        self.assert_norm_s(2, 4)
    
    def test_norm_3s_sto4g(self):
        self.assert_norm_s(3, 4)
    
    def test_norm_4s_sto4g(self):
        self.assert_norm_s(4, 4)
    
    def test_norm_5s_sto4g(self):
        self.assert_norm_s(5, 4)
    

    def test_norm_1s_sto5g(self):
        self.assert_norm_s(1, 5)
    
    def test_norm_2s_sto5g(self):
        self.assert_norm_s(2, 5)
    
    def test_norm_3s_sto5g(self):
        self.assert_norm_s(3, 5)
    
    def test_norm_4s_sto5g(self):
        self.assert_norm_s(4, 5)
    
    def test_norm_5s_sto5g(self):
        self.assert_norm_s(5, 5)
    

    def test_norm_1s_sto6g(self):
        self.assert_norm_s(1, 6)
    
    def test_norm_2s_sto6g(self):
        self.assert_norm_s(2, 6)
    
    def test_norm_3s_sto6g(self):
        self.assert_norm_s(3, 6)
    
    def test_norm_4s_sto6g(self):
        self.assert_norm_s(4, 6)
    
    def test_norm_5s_sto6g(self):
        self.assert_norm_s(5, 6)


    # p-orbitals
    def test_norm_2p_sto1g(self):
        self.assert_norm_p(2, 1)

    def test_norm_3p_sto1g(self):
        self.assert_norm_p(3, 1)
    
    def test_norm_4p_sto1g(self):
        self.assert_norm_p(4, 1)
    
    def test_norm_5p_sto1g(self):
        self.assert_norm_p(5, 1)
    
    
    def test_norm_2p_sto2g(self):
        self.assert_norm_p(2, 2)
    
    def test_norm_3p_sto2g(self):
        self.assert_norm_p(3, 2)
    
    def test_norm_4p_sto2g(self):
        self.assert_norm_p(4, 2)
    
    def test_norm_5p_sto2g(self):
        self.assert_norm_p(5, 2)
    
    
    def test_norm_2p_sto3g(self):
        self.assert_norm_p(2, 3)
    
    def test_norm_3p_sto3g(self):
        self.assert_norm_p(3, 3)
    
    def test_norm_4p_sto3g(self):
        self.assert_norm_p(4, 3)
    
    def test_norm_5p_sto3g(self):
        self.assert_norm_p(5, 3)
    
    
    def test_norm_2p_sto4g(self):
        self.assert_norm_p(2, 4)
    
    def test_norm_3p_sto4g(self):
        self.assert_norm_p(3, 4)
    
    def test_norm_4p_sto4g(self):
        self.assert_norm_p(4, 4)
    
    def test_norm_5p_sto4g(self):
        self.assert_norm_p(5, 4)
    
    
    def test_norm_2p_sto5g(self):
        self.assert_norm_p(2, 5)
    
    def test_norm_3p_sto5g(self):
        self.assert_norm_p(3, 5)
    
    def test_norm_4p_sto5g(self):
        self.assert_norm_p(4, 5)
    
    def test_norm_5p_sto5g(self):
        self.assert_norm_p(5, 5)
    
    
    def test_norm_2p_sto6g(self):
        self.assert_norm_p(2, 6)
    
    def test_norm_3p_sto6g(self):
        self.assert_norm_p(3, 6)
    
    def test_norm_4p_sto6g(self):
        self.assert_norm_p(4, 6)
    
    def test_norm_5p_sto6g(self):
        self.assert_norm_p(5, 6)    

    
    # d-orbitals
    def test_norm_3d_sto1g(self):
        self.assert_norm_d(3, 1)
    
    def test_norm_4d_sto1g(self):
        self.assert_norm_d(4, 1)
    
    def test_norm_5d_sto1g(self):
        self.assert_norm_d(5, 1)
        
    def test_norm_3d_sto2g(self):
        self.assert_norm_d(3, 2)
    
    def test_norm_4d_sto2g(self):
        self.assert_norm_d(4, 2)
    
    def test_norm_5d_sto2g(self):
        self.assert_norm_d(5, 2)
        
    def test_norm_3d_sto3g(self):
        self.assert_norm_d(3, 3)
    
    def test_norm_4d_sto3g(self):
        self.assert_norm_d(4, 3)
    
    def test_norm_5d_sto3g(self):
        self.assert_norm_d(5, 3)
        
    def test_norm_3d_sto4g(self):
        self.assert_norm_d(3, 4)
    
    def test_norm_4d_sto4g(self):
        self.assert_norm_d(4, 4)
    
    def test_norm_5d_sto4g(self):
        self.assert_norm_d(5, 4)
        
    def test_norm_3d_sto5g(self):
        self.assert_norm_d(3, 5)
    
    def test_norm_4d_sto5g(self):
        self.assert_norm_d(4, 5)
    
    def test_norm_5d_sto5g(self):
        self.assert_norm_d(5, 5)
        
    def test_norm_3d_sto6g(self):
        self.assert_norm_d(3, 6)
    
    def test_norm_4d_sto6g(self):
        self.assert_norm_d(4, 6)
    
    def test_norm_5d_sto6g(self):
        self.assert_norm_d(5, 6)

    # f-orbitals
    def test_norm_4f_sto1g(self):
        self.assert_norm_f(4, 1)

    def test_norm_5f_sto1g(self):
        self.assert_norm_f(5, 1)
        
    def test_norm_4f_sto2g(self):
        self.assert_norm_f(4, 2)
    
    def test_norm_5f_sto2g(self):
        self.assert_norm_f(5, 2)
    
    def test_norm_4f_sto3g(self):
        self.assert_norm_f(4, 3)
    
    def test_norm_5f_sto3g(self):
        self.assert_norm_f(5, 3)
        
    def test_norm_4f_sto4g(self):
        self.assert_norm_f(4, 4)
    
    def test_norm_5f_sto4g(self):
        self.assert_norm_f(5, 4)
    
    def test_norm_4f_sto5g(self):
        self.assert_norm_f(4, 5)
    
    def test_norm_5f_sto5g(self):
        self.assert_norm_f(5, 5)
    
    def test_norm_4f_sto6g(self):
        self.assert_norm_f(4, 6)
    
    def test_norm_5f_sto6g(self):
        self.assert_norm_f(5, 6)


    # g-orbitals
    def test_norm_5g_sto1g(self):
        self.assert_norm_g(5, 1)

    def test_norm_5g_sto2g(self):
        self.assert_norm_g(5, 2)
    
    def test_norm_5g_sto3g(self):
        self.assert_norm_g(5, 3)
    
    def test_norm_5g_sto4g(self):
        self.assert_norm_g(5, 4)
    
    def test_norm_5g_sto5g(self):
        self.assert_norm_g(5, 5)
    
    def test_norm_5g_sto6g(self):
        self.assert_norm_g(5, 6)"""
