# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random
from typing import Type

import pytest

import quantumflow as qf


# FIXME; FAILS IF RANGE MAKE LARGER , e.g. (-10, 10)
def random_stdgate(stdgatet: Type[qf.QuantumStdGate]) -> qf.QuantumStdGate:
    """Given a standard gate subclass construct an instance with randomly chosen
    parameters and randomly ordered qubits. Used for testing purposes."""

    args = (random.uniform(-4, 4) for _ in range(0, len(stdgatet.cv_params)))
    qbs = list(range(0, stdgatet.cv_qubit_nb))
    random.shuffle(qbs)
    return stdgatet(*args, *qbs)


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates(gatet: Type[qf.QuantumStdGate]) -> None:

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
def test_stdgates_pow(gatet: Type[qf.QuantumStdGate]) -> None:
    gate0 = random_stdgate(gatet)

    exponent = random.uniform(-4, 4)
    gate1 = gate0 ** exponent
    gate2 = qf.Unitary.from_gate(gate0) ** exponent

    qf.gates_close(gate1, gate2)


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates_structure(gatet: Type[qf.QuantumStdGate]) -> None:
    gate = random_stdgate(gatet)

    if gatet.cv_hermitian:
        assert gate.H is gate


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates_repr(gatet: Type[qf.QuantumStdGate]) -> None:
    gate0 = random_stdgate(gatet)
    rep = repr(gate0)
    gate1 = eval(rep, {gatet.name: gatet for gatet in qf.STDGATES})
    qf.gates_close(gate0, gate1)


def test_stdgates_hash() -> None:
    gate0 = qf.XPow(0.5, 0)
    gate1 = qf.XPow(0.5, 0)
    assert gate0 == gate1

    assert len(set([gate0, gate1])) == 1
