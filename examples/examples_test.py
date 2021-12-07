# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow examples.
"""

import os
import subprocess


def test_prepare_w4_main():
    rval = subprocess.call([os.path.join("examples", "state_prep_w4.py")], shell=True)
    assert rval == 0


def test_prepare_w16_main():
    rval = subprocess.call([os.path.join("examples", "state_prep_w16.py")], shell=True)
    assert rval == 0


def test_prepare_cswap_decomposition():
    rval = subprocess.call(
        [os.path.join("examples", "cswap_decomposition.py")], shell=True
    )
    assert rval == 0


def test_swap_test_main():
    rval = subprocess.call([os.path.join("examples", "swaptest.py")], shell=True)
    assert rval == 0


def test_circuit_identities_main():
    rval = subprocess.call(
        [os.path.join("examples", "circuit_identities.py")], shell=True
    )
    assert rval == 0


def test_gate_translate_identities_main():
    rval = subprocess.call(
        [os.path.join("examples", "gate_translations.py")], shell=True
    )
    assert rval == 0


# fin
