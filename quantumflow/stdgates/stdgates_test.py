# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random
from typing import Type

import pytest

import quantumflow as qf


# FIXME; FAILS IF RANGE MAKE LARGER , e.g. (-10, 10)
def random_stdgate(stdgatet: Type[qf.StdGate]) -> qf.StdGate:
    """Given a standard gate subclass construct an instance with randomly chosen
    parameters and randomly ordered qubits. Used for testing purposes."""

    args = (random.uniform(-4, 4) for _ in range(0, len(stdgatet.cv_params)))
    qbs = list(range(0, stdgatet.cv_qubit_nb))
    random.shuffle(qbs)
    return stdgatet(*args, *qbs)


@pytest.mark.parametrize("name", qf.STDGATES)
def test_stdgates(name: str) -> None:
    gatet = qf.STDGATES[name]

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


@pytest.mark.parametrize("name", qf.STDGATES)
def test_stdgates_pow(name: str) -> None:
    gatet = qf.STDGATES[name]
    gate0 = random_stdgate(gatet)

    exponent = random.uniform(-4, 4)
    gate1 = gate0 ** exponent
    gate2 = qf.Unitary.from_gate(gate0) ** exponent

    qf.gates_close(gate1, gate2)


@pytest.mark.parametrize("name", qf.STDGATES)
def test_stdgates_structure(name: str) -> None:
    gatet = qf.STDGATES[name]
    gate = random_stdgate(gatet)

    if gatet.cv_hermitian:
        assert gate.H is gate

    # TODO: More


@pytest.mark.parametrize("name", qf.STDGATES)
def test_stdgates_repr(name: str) -> None:
    gatet = qf.STDGATES[name]
    gate0 = random_stdgate(gatet)
    rep = repr(gate0)
    gate1 = eval(rep, qf.STDGATES)
    qf.gates_close(gate0, gate1)


def test_stdgates_hash() -> None:
    gate0 = qf.XPow(0.5, 0)
    gate1 = qf.XPow(0.5, 0)
    assert gate0 == gate1

    assert len(set([gate0, gate1])) == 1


@pytest.mark.parametrize("name", qf.STDGATES)
def test_stdgates_hamiltonian(name: str) -> None:
    gate0 = random_stdgate(qf.STDGATES[name])
    qbs = gate0.qubits
    ham = gate0.hamiltonian
    gate1 = qf.Unitary.from_hamiltonian(ham, qbs)
    assert qf.gates_close(gate0, gate1)
