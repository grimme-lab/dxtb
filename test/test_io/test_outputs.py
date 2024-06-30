# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test output.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from dxtb import __version__
from dxtb._src.io.output import (
    get_python_version,
    get_pytorch_version_short,
    get_short_version,
    get_mkl_num_threads,
    get_omp_num_threads,
    get_system_info,
)


@patch("platform.python_version")
def test_python_version(mocker) -> None:
    mocker.return_value = "3.8.5"
    assert get_python_version() == "3.8.5"


@patch("torch.__config__.show")
def test_get_pytorch_version_short(mocker) -> None:
    mocker.return_value = "config,TORCH_VERSION=1.7.1,other"
    assert get_pytorch_version_short() == "1.7.1"


@patch("torch.__config__.show")
def test_get_pytorch_version_short_raises_error(mocker) -> None:
    mocker.return_value = "config,other"

    with pytest.raises(RuntimeError, match="Version string not found in config."):
        get_pytorch_version_short()


@patch("platform.python_version")
@patch("torch.__config__.show")
def test_get_short_version(mocker_torch, mocker_python) -> None:
    mocker_torch.return_value = "config,TORCH_VERSION=1.7.1,other"
    mocker_python.return_value = "3.8.5"

    msg = f"* dxtb version {__version__} running with Python 3.8.5 and PyTorch 1.7.1\n"
    assert get_short_version() == msg


###############################################################################


def test_get_omp_num_threads() -> None:
    # Mock torch.__config__.parallel_info to return a controlled string
    mock_parallel_info = MagicMock()
    mock_parallel_info.return_value = "some_info\nOMP_NUM_THREADS=4\nother_info"

    with patch("torch.__config__.parallel_info", mock_parallel_info):
        omp_num_threads = get_omp_num_threads()
        assert omp_num_threads == "OMP_NUM_THREADS=4"


def test_get_mkl_num_threads() -> None:
    # Mock torch.__config__.parallel_info to return a controlled string
    mock_parallel_info = MagicMock()
    mock_parallel_info.return_value = "some_info\nMKL_NUM_THREADS=8\nother_info"

    with patch("torch.__config__.parallel_info", mock_parallel_info):
        mkl_num_threads = get_mkl_num_threads()
        assert mkl_num_threads == "MKL_NUM_THREADS=8"


def test_get_system_info() -> None:
    with patch("platform.system", return_value="Linux"):
        with patch("platform.machine", return_value="x86_64"):
            with patch("platform.release", return_value="5.4.0-74-generic"):
                with patch("platform.node", return_value="test-host"):
                    with patch("os.cpu_count", return_value=8):
                        system_info = get_system_info()
                        expected_info = {
                            "System Information": {
                                "Operating System": "Linux",
                                "Architecture": "x86_64",
                                "OS Version": "5.4.0-74-generic",
                                "Hostname": "test-host",
                                "CPU Count": 8,
                            }
                        }
                        assert system_info == expected_info
