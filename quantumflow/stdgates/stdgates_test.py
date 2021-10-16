# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random
from typing import Type

import pytest

import quantumflow as qf


# FIXME; FAILS IF RANGE MAKE LARGER , e.g. (-10, 10)
def random_stdgate(gatet: Type[qf.BaseStdGate]) -> qf.BaseStdGate:
    """Given a standard gate subclass construct an instance with randomly chosen
    parameters and randomly ordered qubits. Used for testing purposes."""

    args = (random.uniform(-4, 4) for _ in range(0, len(gatet.cv_params)))
    qbs = list(range(0, gatet.cv_qubit_nb))
    random.shuffle(qbs)
    return gatet(*args, *qbs)  # type: ignore


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates(gatet: Type[qf.BaseStdGate]) -> None:

    # Test creation
    gate0 = random_stdgate(gatet)

    # Test correct number of qubits
    assert gate0.qubit_nb == gatet.cv_qubit_nb

    # Test correct number of arguments
    assert len(gatet.cv_params) == len(gate0.args)

    # Test hermitian conjugate
    assert qf.gates_close(gate0.H, gate0 ** -1)

    # test operator creation
    _ = gate0.sym_operator
    _ = gate0.operator

    # FIXME: more tests


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates_pow(gatet: Type[qf.BaseStdGate]) -> None:
    gate0 = random_stdgate(gatet)

    exponent = random.uniform(-4, 4)
    gate1 = gate0 ** exponent
    gate2 = qf.Unitary.from_gate(gate0) ** exponent

    qf.gates_close(gate1, gate2)


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates_structure(gatet: Type[qf.BaseStdGate]) -> None:
    gate = random_stdgate(gatet)

    if gatet.cv_hermitian:
        assert gate.H is gate
