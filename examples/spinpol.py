import torch

import dxtb
from dxtb import GFN1_XTB, Calculator
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

numbers = torch.tensor([3, 1])  # LiH

spinconst = dxtb.components.spinpolarisation.new_spinpolarisation(numbers, **dd)
print(spinconst.spinconst)

calc2 = dxtb.Calculator(numbers, dxtb.GFN1_XTB, interaction=[spinconst])

# print(spinconst.get_cache)
