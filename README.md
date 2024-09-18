<h1 align="center"><img src="assets/logo.png" width="300"></h3>

<h3 align="center">Fully Differentiable Extended Tight-Binding</h3>
<p align="center">- Combining semi-empirical quantum chemistry with machine learning in PyTorch -</p>

<p align="center">
  <a href="https://github.com/grimme-lab/dxtb/releases/latest">
    <img src="https://img.shields.io/github/v/release/grimme-lab/dxtb?color=orange" alt="Release"/>
  </a>
  <a href="http://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="Apache-2.0"/>
  </a>
  <!---->
  <br>
  <!---->
  <a href="https://github.com/grimme-lab/dxtb/actions/workflows/ubuntu.yaml">
    <img src="https://github.com/grimme-lab/dxtb/actions/workflows/ubuntu.yaml/badge.svg" alt="Test Status Ubuntu"/>
  </a>
  <a href="https://github.com/tgrimme-lab/dxtb/actions/workflows/macos-x86.yaml">
    <img src="https://github.com/grimme-lab/dxtb/actions/workflows/macos-x86.yaml/badge.svg" alt="Test Status macOS (x86)"/>
  </a>
  <a href="https://github.com/tgrimme-lab/dxtb/actions/workflows/macos-arm.yaml">
    <img src="https://github.com/grimme-lab/dxtb/actions/workflows/macos-arm.yaml/badge.svg" alt="Test Status macOS (ARM)"/>
  </a>
  <a href="https://github.com/grimme-lab/dxtb/actions/workflows/windows.yaml">
    <img src="https://github.com/grimme-lab/dxtb/actions/workflows/windows.yaml/badge.svg" alt="Test Status Windows"/>
  </a>
  <!---->
  <br>
  <!---->
  <a href="https://github.com/grimme-lab/dxtb/actions/workflows/release.yaml">
    <img src="https://github.com/grimme-lab/dxtb/actions/workflows/release.yaml/badge.svg" alt="Build Status"/>
  </a>
  <a href="https://dxtb.readthedocs.io">
    <img src="https://readthedocs.org/projects/dxtb/badge/?version=latest" alt="Documentation Status"/>
  </a>
  <a href="https://results.pre-commit.ci/latest/github/grimme-lab/dxtb/main">
    <img src="https://results.pre-commit.ci/badge/github/grimme-lab/dxtb/main.svg" alt="pre-commit.ci Status"/>
  </a>
  <a href="https://codecov.io/gh/grimme-lab/dxtb">
    <img src="https://codecov.io/gh/grimme-lab/dxtb/branch/main/graph/badge.svg?token=O18EZ1CNE3" alt="Coverage"/>
  </a>
  <!---->
  <br>
  <!---->
  <a href="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg">
    <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg" alt="Python Versions"/>
  </a>
  <a href="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg">
    <img src="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg" alt="PyTorch Versions"/>
  </a>
</p>

<br>

The xTB methods (GFNn-xTB) are a series of semi-empirical quantum chemical methods that provide a good balance between accuracy and computational cost.

With *dxtb*, we provide a re-implementation of the xTB methods in PyTorch, which allows for automatic differentiation and seamless integration into machine learning frameworks.

**NOTE**: If you encounter any bugs or have questions on how to use *dxtb*, feel free to open an [issue](https://github.com/grimme-lab/dxtb/issues).


## Installation

### pip <a href="https://pypi.org/project/dxtb/"><img src="https://img.shields.io/pypi/v/dxtb" alt="PyPI Version"></a>

*dxtb* can easily be installed with ``pip``.

```sh
pip install dxtb[libcint]
```

Installing the libcint interface is highly recommended, as it is significantly faster than the pure PyTorch implementation and provides access to higher-order multipole integrals and their derivatives.

### conda <a href="https://anaconda.org/conda-forge/dxtb"><img src="https://img.shields.io/conda/vn/conda-forge/dxtb.svg" alt="Conda Version"></a>


*dxtb* will also be available on [conda](https://conda.io/) soon.

```sh
conda install dxtb
```

Don't forget to install the libcint interface (not on conda) via ``pip install tad-libcint``.

### Other

For more options, see the [installation guide](https://dxtb.readthedocs.io/en/latest/01_quickstart/installation.html) in the documentation.


## Example

The following example demonstrates how to compute the energy and forces using GFN1-xTB.

```python
import torch
import dxtb

dd = {"dtype": torch.double, "device": torch.device("cpu")}

# LiH
numbers = torch.tensor([3, 1], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd)

# instantiate a calculator
calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)

# obtain gradient (dE/dR) via autograd
(g,) = torch.autograd.grad(energy, pos)

# Alternatively, forces can directly be requested from the calculator.
# (Don't forget to manually reset the calculator when the inputs are identical.)
calc.reset()
pos = positions.clone().requires_grad_(True)
forces = calc.get_forces(pos)

assert torch.equal(forces, -g)
```

All quantities are in atomic units.

For more examples and details, check out [the documentation](https://dxtb.readthedocs.io).

## Compatibility

| PyTorch \ Python | 3.8                | 3.9                | 3.10               | 3.11               | 3.12               |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1.11.0           | :white_check_mark: | :white_check_mark: | :x:                | :x:                | :x:                |
| 1.12.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:                |
| 1.13.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.0.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.1.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.2.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| 2.3.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: |
| 2.4.1            | not tested | not tested | not tested | not tested | :x: |

Note that only the latest bug fix version is listed, but all preceding bug fix minor versions are supported.
For example, although only version 2.2.2 is listed, version 2.2.0 and 2.2.1 are also supported.

**Restriction for macOS and Windows:**

On macOS and Windows, PyTorch<2.0.0 does only support Python<3.11.

The libcint interface is **not** available for macOS and Windows.
Correspondingly, the integral evaluation can be considerably slower.
Moreover, higher-order multipole integrals (dipole, quadrupole, ...) are not implemented.
While macOS support may be considered in the future, native Windows support is not possible, because the underlying [libcint](https://github.com/sunqm/libcint) library does not work under Windows.


## Citation

If you use *dxtb* in your research, please cite the following paper:

- M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *dxtb -- An Efficient and Fully Differentiable Framework for Extended Tight-Binding*, *J. Chem. Phys.*, **2024**, 161, 062501. ([DOI](https://doi.org/10.1063/5.0216715))

The Supporting Information can be found [here](https://github.com/grimme-lab/dxtb-data).


For details on the xTB methods, see

- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher, S. Grimme,
  *WIREs Comput. Mol. Sci.*, **2020**, 11, e01493.
  ([DOI](https://doi.org/10.1002/wcms.1493))
- C. Bannwarth, S. Ehlert, S. Grimme,
  *J. Chem. Theory Comput.*, **2019**, 15, 1652-1671.
  ([DOI](https://dx.doi.org/10.1021/acs.jctc.8b01176))
- S. Grimme, C. Bannwarth, P. Shushkov,
  *J. Chem. Theory Comput.*, **2017**, 13, 1989-2009.
  ([DOI](https://dx.doi.org/10.1021/acs.jctc.7b00118))


## Contributing

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
