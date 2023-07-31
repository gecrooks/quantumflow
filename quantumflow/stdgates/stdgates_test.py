# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random
from itertools import chain
from typing import Optional, Type

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.visualization import kwarg_to_symbol


# TODO: Move to gates?
def _randomize_gate(gatet: Type[qf.StdGate]) -> qf.StdGate:
    """Given a StdGate subclass, return an instance with randomly initialized
    parameters and qubits"""
    args = {arg: random.uniform(-4, 4) for arg in gatet.cv_args}

    qubits = list(range(gatet.cv_qubit_nb))
    random.shuffle(qubits)

    for q in range(gatet.cv_qubit_nb):
        args[f"q{q}"] = qubits[q]

    gate = gatet(**args)  # type: ignore

    return gate


def test_str() -> None:
    g0 = qf.Rx(3.12, 2)
    assert str(g0) == "Rx(3.12) 2"

    g1 = qf.H(0)
    assert str(g1) == "H 0"

    g2 = qf.CNot(0, 1)
    assert str(g2) == "CNot 0 1"


def test_repr() -> None:
    g0 = qf.H(0)
    assert repr(g0) == "H(0)"

    g1 = qf.Rx(3.12, 0)
    assert repr(g1) == "Rx(3.12, 0)"

    g2 = qf.CNot(0, 1)
    assert repr(g2) == "CNot(0, 1)"


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_stdgates_repr(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)
    rep = repr(gate0)
    gate1 = eval(rep, dict(qf.STDGATES))
    assert type(gate0) is type(gate1)
    assert qf.gates_close(gate0, gate1)


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_stdgates(gatet: Type[qf.StdGate]) -> None:
    # Test creation
    gate = _randomize_gate(gatet)

    # Test correct number of qubits
    assert gate.qubit_nb == gatet.cv_qubit_nb

    # Test hermitian conjugate
    inv_gate = gate.H
    gate.tensor
    inv_gate.tensor

    # Test inverse
    eye = gate @ inv_gate
    assert qf.gates_close(qf.IdentityGate(range(gate.qubit_nb)), eye)
    assert qf.gates_phase_close(qf.IdentityGate(range(gate.qubit_nb)), eye)

    # Test pow
    assert qf.gates_close(gate**-1, inv_gate)
    assert qf.gates_close((gate**0.5) ** 2, gate)
    assert qf.gates_close((gate**0.3) @ (gate**0.7), gate)

    hgate = qf.Unitary((gate**0.5).tensor, gate.qubits)
    assert qf.gates_close(hgate @ hgate, gate)


def _tensor_structure(tensor: qf.QubitTensor) -> Optional["str"]:
    # "identity", "diagonal", "permutation", "monomial"
    # TODO: Swap

    N = np.ndim(tensor) // 2
    M = np.reshape(tensor, (2**N, 2**N))

    if np.all(M == np.eye(2**N)):
        return "identity"

    if np.all(M == np.diag(np.diagonal(M))):
        return "diagonal"

    if (
        (M.sum(axis=0) == 1).all()
        and (M.sum(axis=0) == 1).all()
        and ((M == 1) | (M == 0)).all()
    ):
        return "permutation"

    A = np.abs(M)
    if (A.sum(axis=0) == A.max(axis=0)).all() and (
        A.sum(axis=1) == A.max(axis=1)
    ).all():
        return "monomial"

    return None


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_tensor_properties(gatet: Type[qf.StdGate]) -> None:
    gate = _randomize_gate(gatet)

    assert qf.almost_unitary(gate)

    if gatet.cv_hermitian:
        assert gate is gate.H
        assert qf.almost_hermitian(gate)

    structure = gatet.cv_tensor_structure

    assert structure in (None, "identity", "diagonal", "permutation", "monomial")

    assert structure == _tensor_structure(gate.tensor)

    if gate.cv_interchangeable:
        qbs = list(gate.qubits)
        random.shuffle(qbs)
        perm_gate = gate.on(*qbs)
        assert qf.gates_close(gate, perm_gate)


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_hamiltonians(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)

    qbs = gate0.qubits
    ham = gate0.hamiltonian

    gate1 = qf.UnitaryGate.from_hamiltonian(ham, qbs)

    assert qf.gates_close(gate0, gate1)
    assert qf.gates_phase_close(gate0, gate1)


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_symbolic(gatet: Type[qf.StdGate]) -> None:
    if len(gatet.cv_args) == 0:
        return

    args = {kwarg_to_symbol[arg]: random.uniform(-4, 4) for arg in gatet.cv_args}
    qbs = range(gatet.cv_qubit_nb)

    # Make gate with symbols
    gate0 = gatet(*args.values(), *qbs)  # type: ignore
    gate1 = gate0.resolve(subs=args)  # type: ignore
    gate2 = gatet(*args.values(), *qbs)  # type: ignore
    assert isinstance(gate1, qf.StdGate)
    assert qf.gates_close(gate1, gate2)

    # Arguments with symbolic constants should be converted to float before
    # creating tensor
    gate0 = gatet(*chain(([qf.PI] * len(args)), qbs))
    gate0.tensor


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_decompose(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)
    circ = qf.Circuit(gate0.decompose())
    gate1 = circ.asgate()
    assert qf.gates_close(gate0, gate1)


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_gate_run(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)

    gate1 = qf.Unitary(gate0.tensor, gate0.qubits)
    ket = qf.random_state(gate0.qubits)

    ket0 = gate0.run(ket)
    ket1 = gate1.run(ket)
    assert qf.states_close(ket0, ket1)


@pytest.mark.parametrize("gatet", qf.STDGATES.values())
def test_gate_evolve(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)

    gate1 = qf.Unitary(gate0.tensor, gate0.qubits)
    rho = qf.random_density(gate0.qubits)

    rho0 = gate0.evolve(rho)
    rho1 = gate1.evolve(rho)
    assert qf.densities_close(rho0, rho1)


def test_diagram_labels() -> None:
    for gatetype in qf.STDGATES.values():
        gate = _randomize_gate(gatetype)
        labels = gate._diagram_labels_()
        assert len(labels) == gate.qubit_nb
        print(f"{gate.name:<16} {str(labels):<50}")


@pytest.mark.parametrize("gatetype", qf.STDCTRLGATES.values())
def test_StdCtrlGate_tensor_structure(gatetype: Type[qf.StdCtrlGate]) -> None:
    struct = gatetype.cv_target.cv_tensor_structure

    # TODO: make this automatic, rather than testing?

    if struct == "swap":
        assert gatetype.cv_tensor_structure == "permutation"
    else:
        assert gatetype.cv_target.cv_tensor_structure == gatetype.cv_tensor_structure

    assert gatetype.cv_hermitian == gatetype.cv_target.cv_hermitian


# fin
