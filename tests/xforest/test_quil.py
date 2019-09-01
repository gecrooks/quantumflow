
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest
pytest.importorskip("pyquil")      # noqa: 402

from quantumflow import xforest

QUIL_FILES = [
    'hello_world.quil',
    'empty.quil',
    # 'classical_logic.quil',   # Needs to declare classical data?
    # 'control_flow.quil',      # Needs to declare classical data?
    'measure.quil',
    'qaoa.quil',
    'bell.quil',
    # 'include.quil',
    ]

RUNNABLE_QUIL_FILES = QUIL_FILES[:-1]

QUILDIR = 'tests/xforest/quil/'


def test_parse_quilfile():
    print()
    for quilfile in QUIL_FILES:
        filename = QUILDIR+quilfile
        print("<<<"+filename+">>>")
        with open(filename, 'r') as f:
            quil = f.read()
        xforest.quil_to_program(quil)


def test_run_quilfile():
    print()
    for quilfile in RUNNABLE_QUIL_FILES:
        filename = QUILDIR+quilfile
        print("<<<"+filename+">>>")
        with open(filename, 'r') as f:
            quil = f.read()
        prog = xforest.quil_to_program(quil)
        prog.run()


def test_unparsable():
    with pytest.raises(RuntimeError):
        filename = QUILDIR + 'unparsable.quil'
        with open(filename, 'r') as f:
            quil = f.read()
        xforest.quil_to_program(quil)
