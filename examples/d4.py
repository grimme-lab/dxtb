import torch

from dxtb import GFN2_XTB, Calculator

numbers = torch.tensor([3, 1])
positions = torch.tensor([[0.0, 0.0, -1.508], [0.0, 0.0, 1.508]])
# float32 singlepoint without spin
calc = Calculator(
    numbers, par=GFN2_XTB, device=torch.device("cpu"), dtype=torch.float
)
try:
    result = calc.singlepoint(positions)
    print("float32 works:", result.total.sum().item())
except RuntimeError as e:
    print("float32 fails:", e)
