# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
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
import argparse
import sys

from .modes import enable_debug


def parse_args():
    # parse the argument
    parser = argparse.ArgumentParser("Run python script by enabling xitorch debug mode")
    parser.add_argument("scriptfile", type=str, help="Path to the script to run")
    parser.add_argument(
        "args",
        type=str,
        nargs=argparse.REMAINDER,
        help="The arguments needed to run the script",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scriptfile = args.scriptfile
    scriptargs = args.args if args.args is not None else []
    scriptargs.insert(0, scriptfile)
    sys.argv[:] = scriptargs[:]

    # compile and run the code with debug mode enabled
    with enable_debug():
        with open(scriptfile, "rb") as stream:
            code = compile(stream.read(), scriptfile, "exec")
        globs = {
            "__file__": scriptfile,
            "__name__": "__main__",
            "__package__": None,
            "__cached__": None,
        }
        exec(code, globs, None)


if __name__ == "__main__":
    main()
