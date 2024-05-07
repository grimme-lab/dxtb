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
SCF: Implicit
=============

SCF implementations that utilize the implicit function theorem in the backward
pass, i.e., using a closed form expression for the gradient instead of
unrolling all iterations. This is inherently more memory efficient (constant
memory), but holds some caveats for the AD engine.

Note
----
Currently, the implicit SCF implementations are not fully compatible with
PyTorch's composable function transforms.
"""
from .default import SelfConsistentFieldImplicit
