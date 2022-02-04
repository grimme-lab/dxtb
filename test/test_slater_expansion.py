
###################################
import sys 
import os
# Add the src directory to sys.path so that all imports in the unittests work
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../src/")
sys.path.insert(0, src_directory)
################################### # TODO: remove

from unittest import TestCase
import numpy as np
import math
import sys

from basis import Cgto_Type
from slater import slater_to_gauss
from overlap import overlap_cgto

""" Testing the functionality of the overlap. """

'''# TODO: check
def epsilon(x: float):
    return x + sys.float_info.epsilon
thr = 5e+6*epsilon(1.0)
thr2 = math.sqrt(epsilon(1.0))'''

# equality threshold
thr_atol = 1e-07


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

        # same site 
        r2 = 0.0
        vec = np.array([0., 0., 0.])

        # create gaussians
        cgto = Cgto_Type()
        slater_to_gauss(ng, n, l, 1.0, cgto, True)

        # calculate self-overlap
        overlap = overlap_cgto(cgto, cgto, r2, vec, 100.0)

        # self-overlap should be identity matrix
        target = np.identity(dim)
        self.assertTrue(np.allclose(overlap, target, rtol=1e-05, atol=thr_atol, equal_nan=False), msg=f"Self overlap not identity:\n {overlap}")

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
    
    def test_norm_2s_sto1g(self):
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
        self.assert_norm_g(5, 6)




if __name__ == "__main__":
    print("Start debugging")
    tse = Test_Slater_Expansion()
    #tse.assert_norm_s(1,1)
    #tse.assert_norm_p(2,1)
    #tse.assert_norm_p(5,6)
    #tse.assert_norm_d(3,2)
    tse.assert_norm_f(4,5)
