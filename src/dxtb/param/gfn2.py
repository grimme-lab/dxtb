# This file is part of xtbml.
from __future__ import annotations

import os.path as op

import tomli as toml

from .base import Param

with open(op.join(op.dirname(__file__), "gfn2-xtb.toml"), "rb") as fd:
    GFN2_XTB = Param(**toml.load(fd))
