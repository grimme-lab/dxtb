"""
Setup for pytest.
"""

import pytest
import torch

torch.set_printoptions(precision=10)


def pytest_addoption(parser: pytest.Parser):
    """Set up additional command line options."""

    parser.addoption(
        "--detect-anomaly",
        action="store_true",
        help="Enable more comprehensive gradient tests.",
    )


def pytest_configure(config: pytest.Config):
    """Pytest configuration hook."""

    if config.getoption("--detect-anomaly"):
        torch.autograd.anomaly_mode.set_detect_anomaly(True)
