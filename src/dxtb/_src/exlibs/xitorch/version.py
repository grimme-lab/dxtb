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
import os
import subprocess as sp

__all__ = ["get_version"]

MAJOR = 0
MINOR = 4
MICRO = 0
ISRELEASED = False
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
# taken from numpy/numpy
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def _get_git_version():
    cwd = os.getcwd()

    # go to the main directory
    fdir = os.path.dirname(os.path.abspath(__file__))
    maindir = os.path.abspath(os.path.join(fdir, ".."))
    # maindir = fdir # os.path.join(fdir, "..")
    os.chdir(maindir)

    # get git version
    res = git_version()

    # restore the cwd
    os.chdir(cwd)
    return res


def get_version(build_version=False):
    if ISRELEASED:
        return VERSION

    # unreleased version
    GIT_REVISION = _get_git_version()
    if build_version:
        import datetime as dt

        date = dt.date.strftime(dt.datetime.now(), "%Y%m%d%H%M%S")
        return VERSION + ".dev" + date
    else:
        return VERSION + ".dev0+" + GIT_REVISION[:7]
