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
Configuration
=============

This module contains the configuration classes for the :class:`dxtb.Calculator`.

For simplicity, all options can be passed to the :class:`~dxtb.Calculator` in a
simple dictionary, from which the :class:`~dxtb.config.Config` object is
created.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])
    opts = {"maxiter": 100}
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts)

The :class:`~dxtb.Calculator` stores the configuration in the
:attr:`~dxtb.Calculator.opts` attribute.
All options can be accessed and modified directly using the attribute.

Note that the options are passed to separate configuration classes within the
main configuration class. The maximum number of SCF iterations is passed to the
:class:`~dxtb.config.ConfigSCF` and can be accessed as follows:

.. code-block:: python

    print(calc.opts.scf.maxiter)

For all available options, see the documentation of the configuration classes.
The defaults for the configuration classes can be found there. Additionally,
the defaults are stored in :mod:`~dxtb._src.constants.defaults`.

Verbosity
---------

The only option that is not passed to a configuration class is the verbosity
level. This option is passed directly to the :class:`~dxtb.OutputHandler`.

The default verbosity level is set to 5. The minimum verbosity level is 0,
which corresponds to no output. The maximum verbosity level is 10.
"""
from .cache import *
from .integral import *
from .main import *
from .scf import *
