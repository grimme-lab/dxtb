"""
Setup for pytest.
"""

import pytest
import torch

# avoid randomness and non-deterministic algorithms
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

torch.set_printoptions(precision=10)


def pytest_addoption(parser: pytest.Parser):
    """Set up additional command line options."""

    parser.addoption(
        "--detect-anomaly",
        action="store_true",
        help="Enable more comprehensive gradient tests.",
    )

    parser.addoption(
        "--tpo-linewidth",
        action="store",
        default=400,
        type=int,
        help=(
            "The number of characters per line for the purpose of inserting "
            "line breaks (default = 80). Thresholded matrices will ignore "
            "this parameter."
        ),
    )

    parser.addoption(
        "--tpo-precision",
        action="store",
        default=6,
        type=int,
        help=(
            "Number of digits of precision for floating point output " "(default = 4)."
        ),
    )

    parser.addoption(
        "--tpo-threshold",
        action="store",
        default=1000,
        type=int,
        help=(
            "Total number of array elements which trigger summarization "
            "rather than full `repr` (default = 1000)."
        ),
    )


def pytest_configure(config: pytest.Config):
    """Pytest configuration hook."""

    if config.getoption("--detect-anomaly"):
        torch.autograd.anomaly_mode.set_detect_anomaly(True)

    if config.getoption("--tpo-linewidth"):
        torch.set_printoptions(linewidth=config.getoption("--tpo-linewidth"))

    if config.getoption("--tpo-precision"):
        torch.set_printoptions(precision=config.getoption("--tpo-precision"))

    if config.getoption("--tpo-threshold"):
        torch.set_printoptions(threshold=config.getoption("--tpo-threshold"))

    # register an additional marker
    config.addinivalue_line("markers", "cuda: mark test that require CUDA.")


def pytest_runtest_setup(item: pytest.Function):
    """Custom marker for tests requiring CUDA."""

    for _ in item.iter_markers(name="cuda"):
        if not torch.cuda.is_available():
            pytest.skip(
                "Torch not compiled with CUDA enabled or no CUDA device available."
            )
