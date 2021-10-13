# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random
from typing import Type

import numpy as np
import pytest

import quantumflow as qf


def random_stdgate(gatet: Type[qf.BaseStdGate]) -> qf.BaseStdGate:
    """Given a standard gate subclass construct an instance with randomly chosen
    parameters and randomly ordered qubits. Used for testing purposes."""

    args = (random.uniform(-10, 10) for _ in range(0, len(gatet.cv_params)))
    qbs = list(range(0, gatet.cv_qubit_nb))
    random.shuffle(qbs)
    return gatet(*args, *qbs)  # type: ignore


def test_Rx() -> None:
    assert qf.Rx in qf.OPERATIONS
    assert qf.Rx in qf.GATES
    assert qf.Rx in qf.STDGATES
    assert qf.Rx not in qf.STDCTRLGATES

    gate0 = qf.Rx(0.2, 1)

    assert gate0.name == "Rx"
    assert gate0.qubits == (1,)
    assert gate0.qubit_nb == 1
    assert gate0.cbits == ()
    assert gate0.cbit_nb == 0
    assert gate0.qubit_nb == gate0.cv_qubit_nb
    assert gate0.args == (0.2,)
    assert gate0.cv_operator_structure == qf.OperatorStructure.unstructured
    assert gate0.asgate() is gate0

    # FIXME More tests
    _ = gate0.sym_operator


def test_CX() -> None:
    gate0 = qf.CX(1, 4)

    assert gate0.control_qubits == (1,)
    assert gate0.control_qubit_nb == 1
    assert isinstance(gate0.target, qf.X)
    assert gate0.target.qubits == (4,)
    assert gate0.target.qubit_nb == 1

    # FIXME: More tests


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates(gatet: Type[qf.BaseStdGate]) -> None:

    # Test creation
    gate0 = random_stdgate(gatet)

    # Test correct number of qubits
    assert gate0.qubit_nb == gatet.cv_qubit_nb

    # Test correct number of arguments
    assert len(gatet.cv_params) == len(gate0.args)

    # Test hermitian conjugate
    gate1 = gate0.H
    assert np.allclose(gate1.operator, gate0.operator.conj().T)

    # Test inversion
    _ = gate0 ** -1
    # FIXME: more tests


@pytest.mark.parametrize("gatet", qf.STDGATES)
def test_stdgates_structure(gatet: Type[qf.BaseStdGate]) -> None:
    gate = random_stdgate(gatet)

    if gatet.cv_hermitian:
        assert gate.H is gate
