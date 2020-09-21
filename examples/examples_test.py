# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow examples.
"""

import subprocess

import numpy as np

import quantumflow as qf


def test_prepare_w4():
    import examples.state_prep_w4 as ex

    ket = ex.prepare_w4()
    assert qf.states_close(ket, qf.w_state(4))


def test_prepare_w4_main():
    rval = subprocess.call(["examples/state_prep_w4.py"])
    assert rval == 0


def test_prepare_w16():
    import examples.state_prep_w16 as ex

    ket = ex.prepare_w16()
    assert qf.states_close(ket, qf.w_state(16))


def test_prepare_w16_main():
    rval = subprocess.call(["examples/state_prep_w16.py"])
    assert rval == 0


def test_prepare_cswap_decomposition():
    rval = subprocess.call(["examples/cswap_decomposition.py"])
    assert rval == 0


def test_swap_test():
    import examples.swaptest as ex

    ket0 = qf.zero_state([0])
    ket1 = qf.random_state([1])
    ket2 = qf.random_state([2])

    ket = qf.join_states(ket0, ket1)
    ket = qf.join_states(ket, ket2)

    fid = qf.state_fidelity(ket1, ket2.on(1))
    st_fid = ex.swap_test(ket, 0, 1, 2)

    assert np.isclose(fid, st_fid)


def test_swap_test_main():
    rval = subprocess.call(["examples/swaptest.py"])
    assert rval == 0


def test_circuit_identities_main():
    rval = subprocess.call(["examples/circuit_identities.py"])
    assert rval == 0


def test_gate_translate_identities_main():
    rval = subprocess.call(["examples/gate_translations.py"])
    assert rval == 0


# def test_fit_state():
#     from examples import fit_state
#     fit_state.example_fit_state()

# fin
