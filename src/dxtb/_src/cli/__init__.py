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
Command Line Interface
======================

``dxtb`` also provides a command line interface (CLI) to run some basic
calculations.

The only required argument is the path to the input file.

.. code-block:: bash

    dxtb mol.xyz

Some important options are listed below:

- ``--forces``: Calculate forces.
- ``--dipole``: Calculate dipole moment.
- ``--verbosity <int>``: Set verbosity level. Also use ``-v`` to increase and
  ``-s`` to decrease; can be used multiple times (e.g., ``-vv``, ``-sss``).
- ``--device <device>``: Device for calculations.

For all available options, run:

.. code-block:: bash

    dxtb --help
"""

from .argparser import parser
from .driver import Driver
from .entrypoint import console_entry_point
