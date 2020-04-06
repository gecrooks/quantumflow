
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.meta
"""

import io
import subprocess
import glob

from quantumflow import meta


def test_print_versions():
    out = io.StringIO()
    meta.print_versions(out)
    print(out)


def test_print_versions_main():
    rval = subprocess.call(['python', '-m', 'quantumflow.meta'])
    assert rval == 0


# TODO: Make test more specific for complete header
# TODO: Include tests, examples, tools, ...
def test_copyright():
    """Check that source code files contain copyright line"""
    exclude = set(['quantumflow/version.py', 'quantumflow/__init__.py'])
    for fname in glob.glob('quantumflow/**/*.py', recursive=True):
        if fname in exclude:
            continue
        print(fname)

        with open(fname) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                assert line.startswith('# Copyright')
                break
