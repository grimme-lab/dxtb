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
Information about the system, settings and PyTorch.
"""

from __future__ import annotations

import os
import platform

import torch

from dxtb.__version__ import __tversion__

__all__ = [
    "get_mkl_num_threads",
    "get_omp_num_threads",
    "get_pytorch_info",
    "get_system_info",
    "print_system_info",
]


def get_omp_num_threads() -> str:
    omp_num_threads = "1"

    parinfo = torch.__config__.parallel_info().split("\n")
    for info in parinfo:
        if "OMP_NUM_THREADS" in info:
            omp_num_threads = info.split()[-1]

    return omp_num_threads


def get_mkl_num_threads() -> str:
    mkl_num_threads = "1"

    parinfo = torch.__config__.parallel_info().split("\n")
    for info in parinfo:
        if "MKL_NUM_THREADS" in info:
            mkl_num_threads = info.split()[-1]

    return mkl_num_threads


def get_system_info():
    return {
        "System Information": {
            "Operating System": platform.system(),
            "Architecture": platform.machine(),
            "OS Version": platform.release(),
            "Hostname": platform.node(),
            "CPU Count": os.cpu_count(),
        }
    }


def get_pytorch_info():
    is_cuda = torch.cuda.is_available()

    backends = []
    parallel = []

    config = torch.__config__.show().split("\n")
    for info in config:
        if "OpenMP" in info:
            if not torch.backends.openmp.is_available():  # type: ignore
                raise RuntimeError(
                    "PyTorch build against OpenMP, but not available from "
                    "`torch.backends.openmp`."
                )
            openmp = info.replace(")", "").replace("(", "").split()[-1]
            parallel.append(f"OpenMP {openmp}")

        if "Build settings" in info:
            for i in info.strip().split(","):
                if "BLAS_INFO" in i:
                    blas = i.strip().split("=")[-1]
                    backends.append(f"BLAS={blas}")
                if "LAPACK_INFO" in i:
                    lapack = i.strip().split("=")[-1]
                    backends.append(f"LAPACK={lapack}")
                if "USE_MPI" in i:
                    if i.strip().split("=")[-1] == "ON":
                        parallel.append("MPI")
                if "USE_ROCM" in i:
                    if i.strip().split("=")[-1] == "ON":
                        backends.append("ROCM")

    cuda_version = (
        "None"
        if not is_cuda
        else f"{torch.version.cuda} ({torch._C._cuda_getCompiledVersion()})"  # type: ignore
    )

    cuda_devices = (
        "None"
        if not is_cuda
        else f"{torch.cuda.device_count()} (default: {torch.cuda.get_device_name(0)})"
    )

    return {
        "PyTorch Information": {
            "PyTorch Version": torch.__version__,
            "CUDA": cuda_version,
            "CUDA Devices": cuda_devices,
            "Default Device": str(torch.tensor(0.0).device),
            "Default Dtype": str(torch.tensor(0.0).dtype),
            "CPU Parallelism": parallel,
            "OMP_NUM_THREADS": get_omp_num_threads(),
            "MKL_NUM_THREADS": get_mkl_num_threads(),
            "Backends": backends,
        }
    }


def print_system_info(punit=print):
    system_info = get_system_info()["System Information"]
    pytorch_info = get_pytorch_info()["PyTorch Information"]
    sep = 17

    punit("")
    punit("System Information")
    punit("------------------")
    punit("")

    # OS
    punit(
        f"{'Operating System'.ljust(sep)}: {system_info['Operating System']} "
        f"({system_info['OS Version']})"
    )
    punit(f"{'Architecture'.ljust(sep)}: {system_info['Architecture']} ")
    punit(f"{'Hostname'.ljust(sep)}: {system_info['Hostname']} ")
    punit(f"{'CPU Count'.ljust(sep)}: {system_info['CPU Count']} ")

    # PyTorch
    punit("")
    punit(f"{'PyTorch Version'.ljust(sep)}: {pytorch_info['PyTorch Version']}")
    punit(f"{'CUDA'.ljust(sep)}: {pytorch_info['CUDA']}")
    punit(f"{'CUDA Devices'.ljust(sep)}: {pytorch_info['CUDA Devices']}")
    punit(f"{'Default Device'.ljust(sep)}: {pytorch_info['Default Device']}")
    punit(f"{'Default Dtype'.ljust(sep)}: {pytorch_info['Default Dtype']}")
    punit(f"{'CPU Parallelism'.ljust(sep)}: {pytorch_info['CPU Parallelism']}")
    punit(f"{'OMP_NUM_THREADS'.ljust(sep)}: {pytorch_info['OMP_NUM_THREADS']}")
    punit(f"{'MKL_NUM_THREADS'.ljust(sep)}: {pytorch_info['MKL_NUM_THREADS']}")
    punit(f"{'Backends'.ljust(sep)}: {pytorch_info['Backends']}")
    punit("")
