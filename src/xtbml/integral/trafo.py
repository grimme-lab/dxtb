""" Integral transformation """

import math
import torch

from ..exceptions import IntegralTransformError


dtype = torch.float64

# fmt: off
s3 = math.sqrt(3.0)
s3_4 = s3 * 0.5
dtrafo = torch.tensor([
      #   0       1      -1       2      -2
      -0.5, 0.0, 0.0,   s3_4, 0.0,  # xx
      -0.5, 0.0, 0.0,  -s3_4, 0.0,  # yy
       1.0, 0.0, 0.0, 0.0, 0.0,  # zz
       0.0, 0.0, 0.0, 0.0,     s3,  # xy
       0.0,     s3, 0.0, 0.0, 0.0,  # xz
       0.0, 0.0,     s3, 0.0, 0.0], # yz
       dtype=dtype).reshape(6, 5)
d32 = 3.0/2.0
s3_8 = math.sqrt(3.0/8.0)
s5_8 = math.sqrt(5.0/8.0)
s6 = math.sqrt(6.0)
s15 = math.sqrt(15.0)
s15_4 = math.sqrt(15.0/4.0)
s45 = math.sqrt(45.0)
s45_8 = math.sqrt(45.0/8.0)
ftrafo = torch.tensor([
      # 0        1       -1         2       -2         3       -3 
       0.0,   -s3_8,  0.0,   0.0,  0.0,     s5_8,  0.0,  # xxx 
       0.0,  0.0,   -s3_8,   0.0,  0.0,   0.0,   -s5_8,  # yyy 
       1.0,  0.0,  0.0,   0.0,  0.0,   0.0,  0.0,  # zzz 
       0.0,  0.0,   -s3_8,   0.0,  0.0,   0.0,   s45_8,  # xxy 
         -d32,  0.0,  0.0,    s15_4,  0.0,   0.0,  0.0,  # xxz 
       0.0,   -s3_8,  0.0,   0.0,  0.0,   -s45_8,  0.0,  # xyy 
         -d32,  0.0,  0.0,   -s15_4,  0.0,   0.0,  0.0,  # yyz 
       0.0,      s6,  0.0,   0.0,  0.0,   0.0,  0.0,  # xzz 
       0.0,  0.0,      s6,   0.0,  0.0,   0.0,  0.0,  # yzz 
       0.0,  0.0,  0.0,   0.0,     s15,   0.0,  0.0], # xyz 
       dtype=dtype).reshape(10, 7)

d38 = 3.0/8.0
d34 = 3.0/4.0
s5_16 = math.sqrt(5.0/16.0)
s10 = math.sqrt(10.0)
s10_8 = math.sqrt(10.0/8.0)
s35_4 = math.sqrt(35.0/4.0)
s35_8 = math.sqrt(35.0/8.0)
s35_64 = math.sqrt(35.0/64.0)
s45_4 = math.sqrt(45.0/4.0)
s315_8 = math.sqrt(315.0/8.0)
s315_16 = math.sqrt(315.0/16.0)
gtrafo = torch.tensor([
      #    0       1      -1       2      -2        3      -3         4      -4
          d38, 0., 0.,-s5_16, 0.,  0., 0.,  s35_64, 0.,  # xxxx
          d38, 0., 0., s5_16, 0.,  0., 0.,  s35_64, 0.,  # yyyy
        1., 0., 0., 0., 0.,  0., 0.,   0., 0.,  # zzzz
        0., 0., 0., 0.,-s10_8,  0., 0.,   0., s35_4,  # xxxy
        0.,-s45_8, 0., 0., 0.,  s35_8, 0.,   0., 0.,  # xxxz
        0., 0., 0., 0.,-s10_8,  0., 0.,   0.,-s35_4,  # xyyy
        0., 0.,-s45_8, 0., 0.,  0.,-s35_8,   0., 0.,  # yyyz
        0.,   s10, 0., 0., 0.,  0., 0.,   0., 0.,  # xzzz
        0., 0.,   s10, 0., 0.,  0., 0.,   0., 0.,  # yzzz
          d34, 0., 0., 0., 0.,  0., 0.,-s315_16, 0.,  # xxyy
       -3., 0., 0., s45_4, 0.,  0., 0.,   0., 0.,  # xxzz
       -3., 0., 0.,-s45_4, 0.,  0., 0.,   0., 0.,  # yyzz
        0., 0.,-s45_8, 0., 0.,  0.,s315_8,   0., 0.,  # xxyz
        0.,-s45_8, 0., 0., 0.,-s315_8, 0.,   0., 0.,  # xyyz
        0., 0., 0., 0.,   s45,  0., 0.,   0., 0.], # xyzz
       dtype=dtype).reshape(15, 9)
# fmt: on


def transform0(lj: int, li: int, cart):
    """Transform overlap. Note the inverted order of li, lj.


    Args:
        lj (int): [description]
        li (int): [description]
        cart ([type]): Cartesian coordinates

    Raises:
        IntegralTransformError: [description]

    Returns:
        [type]: [description]
    """  # TODOC

    # infer shape depending on l
    dim = cart.shape[0] - 1
    sphr = torch.zeros((dim, dim))
    assert len(cart.shape) == 2

    if li == 0 or li == 1:
        if lj == 0 or lj == 1:
            sphr = cart
        elif lj == 2:
            sphr = torch.matmul(cart, dtrafo)
        elif lj == 3:
            sphr = torch.matmul(cart, ftrafo)
        elif lj == 4:
            sphr = torch.matmul(cart, gtrafo)
        else:
            raise IntegralTransformError
    elif li == 2:
        if lj == 0 or lj == 1:
            sphr = torch.matmul(torch.transpose(dtrafo, 0, 1), cart)
        elif lj == 2:
            sphr = torch.matmul(
                torch.transpose(dtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), dtrafo),
            )
        elif lj == 3:
            sphr = torch.matmul(
                torch.transpose(ftrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), dtrafo),
            )
        elif lj == 4:
            sphr = torch.matmul(
                torch.transpose(gtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), dtrafo),
            )
        else:
            raise IntegralTransformError
    elif li == 3:
        if lj == 0 or lj == 1:
            sphr = torch.matmul(torch.transpose(ftrafo, 0, 1), cart)
        elif lj == 2:
            sphr = torch.matmul(
                torch.transpose(dtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), ftrafo),
            )
        elif lj == 3:
            sphr = torch.matmul(
                torch.transpose(ftrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), ftrafo),
            )
        elif lj == 4:
            sphr = torch.matmul(
                torch.transpose(gtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), ftrafo),
            )
        else:
            raise IntegralTransformError
    elif li == 4:
        if lj == 0 or lj == 1:
            sphr = torch.matmul(torch.transpose(gtrafo, 0, 1), cart)
        elif lj == 2:
            sphr = torch.matmul(
                torch.transpose(dtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), gtrafo),
            )
        elif lj == 3:
            sphr = torch.matmul(
                torch.transpose(ftrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), gtrafo),
            )
        elif lj == 4:
            sphr = torch.matmul(
                torch.transpose(gtrafo, 0, 1),
                torch.matmul(torch.transpose(cart, 0, 1), gtrafo),
            )
        else:
            raise IntegralTransformError
    else:
        raise IntegralTransformError

    return sphr


def transform1(lj: int, li: int, cart, sphr):

    assert len(cart.shape) == 3
    assert len(sphr.shape) == 3

    for k in range(torch.size(cart, 0)):
        sphr[k, :, :] = transform0(lj, li, cart[k, :, :], sphr[k, :, :])

    return sphr


def transform2(lj: int, li: int, cart, sphr):

    assert len(cart.shape) == 4
    assert len(sphr.shape) == 4

    for l in range(torch.size(cart, 1)):
        for k in range(torch.size(cart, 0)):
            sphr[k, l, :, :] = transform0(lj, li, cart[k, l, :, :], sphr[k, l, :, :])

    return sphr
