import torch

# SCF settings
ETEMP = 300
MAXITER = 20
VERBOSITY = 1

# Convergence thresholds
THRESH = {
    torch.float16: torch.tensor(1e-2, dtype=torch.float16),
    torch.float32: torch.tensor(1e-5, dtype=torch.float32),
    torch.float64: torch.tensor(1e-10, dtype=torch.float64),
}
