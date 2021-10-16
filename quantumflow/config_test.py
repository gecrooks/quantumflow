# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.

import glob
import io
import subprocess

import quantumflow as qf

# DOCME
REPS = 16


def test_version() -> None:
    assert qf.__version__


def test_about() -> None:
    out = io.StringIO()
    qf.about(out)
    print(out)


def test_about_main() -> None:
    rval = subprocess.call(["python", "-m", "quantumflow.about"])
    assert rval == 0


def test_copyright() -> None:
    """Check that source code files contain a copyright line"""
    for fname in glob.glob("quantumflow/**/*.py", recursive=True):
        print("Checking " + fname + " for copyright header")

        with open(fname) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                assert line.startswith("# " + qf.__copyright__)
                break
