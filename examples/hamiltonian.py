import time

import torch
from tad_mctc.typing import DD

from dxtb.basis import IndexHelper
from dxtb.integral import Hamiltonian
from dxtb.param import GFN1_XTB as par
from dxtb.param.util import get_elem_angular

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

numbers = torch.randint(1, 86, (50,))
print(numbers)
ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
hcore = Hamiltonian(numbers, par, ihelp, **dd)


def time_function(func, repeats: int, *args, **kwargs) -> tuple[float, float]:
    """
    Time the execution of a function over a specified number of repeats.

    Parameters:
    - func: The function to time.
    - repeats: The number of times to execute the function.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - Tuple containing the total execution time and average execution time.
    """
    total_time = 0.0

    for _ in range(repeats):
        start_time = time.time()
        func(*args, **kwargs)  # Execute the function with provided arguments
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / repeats

    print(f"Total {func.__name__}: {total_time:.4f} seconds")
    print(f"Average {func.__name__}: {average_time:.4f} seconds")
    return total_time, average_time


REPS = 5
time_function(hcore.integral._get_hscale, REPS)


a1 = hcore.integral._get_hscale()
print((a1 == a1.mT).all())
