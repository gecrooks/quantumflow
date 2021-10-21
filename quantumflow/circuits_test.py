# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import pytest

import quantumflow as qf


def test_Circuit_init() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.S(2), qf.T(1))
    assert circ0.qubits == (0, 1, 2)
    assert len(circ0) == 3

    circ1 = circ0.H
    assert qf.gates_close(circ1.asgate().H, circ0.asgate())

    qbs = (2, 1, 0, 4, 5)
    circ2 = qf.Circuit(qf.X(0), qf.S(2), qf.T(1), qubits=qbs)
    assert circ2.qubits == qbs
    assert circ2.H.qubits == qbs

    with pytest.raises(ValueError):
        _ = qf.Circuit(qf.X(10), qf.S(2), qf.T(1), qubits=qbs)

    assert qf.X(0) in circ0
    assert qf.X(1) not in circ0


def test_Circuit_add() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.S(2), qf.T(1))

    circ0 += qf.H(3)
    circ4 = circ0 + (qf.X(0), qf.X(1))
    assert len(circ4) == 6
    assert circ4.qubits == (0, 1, 2, 3)

    circ5 = circ4 + circ4
    circ6 = circ5.add(circ4)
    assert len(circ6) == 18

    circ2 = qf.Circuit(qf.X(0), qubits=(0, 5))
    circ2 += qf.S(5)

    circ7 = circ6[3:6]
    assert isinstance(circ7, qf.Circuit)
    assert len(circ7) == 3

    op = circ6[1]
    assert isinstance(op, qf.S)


def test_Circuit_on() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Y(2), qf.Z(1))
    circ1 = circ0.on(["a", "c", "b"])
    assert circ1.qubits == ("a", "c", "b")
    assert circ1[1].qubits == ("b",)

    circ0 = qf.Circuit(qf.X(0), qf.Y(2), qf.Z(1), qubits=[0, 2, 1])
    circ1 = circ0.on(["a", "c", "b"])
    assert circ1.qubits == ("a", "c", "b")
    assert circ1[1].qubits == ("c",)

    with pytest.raises(ValueError):
        _ = circ0.on(["a", "c", "b", "d"])


def test_Circuit_relabel() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Y(2), qf.Z(1))
    circ1 = circ0.relabel({0: "a", 1: "b", 2: "c", 3: "d"})
    assert circ1[1].qubits == ("c",)
    assert circ1.qubits == ("a", "b", "c", "d")


def test_Circuit_flat() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Y(0), qf.Z(0))
    circ1 = qf.Circuit(qf.H(0), circ0, qf.H(0))

    assert len(circ1) == 3
    circ_flat = qf.Circuit(*circ1.flat())
    assert len(circ_flat) == 5

    circ3 = qf.Circuit(*[circ0] * 4)
    circ4 = qf.Circuit(*circ3.flat())
    assert len(circ4) == len(circ3) * len(circ0)


def test_Moment() -> None:
    moment = qf.Moment(qf.X(0), qf.S(2), qf.T(1))
    assert moment.qubits == (0, 1, 2)

    with pytest.raises(ValueError):
        _ = qf.Moment(qf.X(0), qf.S(2), qf.T(2))


def test_DAGCircuit_depth() -> None:
    dagc = qf.DAGCircuit(
        qf.I(0),
        qf.X(0),
        qf.Y(0),
        qf.Z(0),
        qf.S(1),
        qf.T(1),
    )

    assert len(dagc) == 6
    assert dagc.depth() == 4
    assert dagc.component_nb() == 2

    comps = dagc.components()
    assert len(comps) == 2
    assert len(comps[0]) == 4
    assert len(comps[1]) == 2

    moments = list(dagc.moments())
    assert len(moments) == 4

    assert qf.I(0) in dagc
    assert qf.I(2) not in dagc

    assert dagc.qubits == (0, 1)


def test_Circuit_eq() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Y(2))
    assert circ0 == qf.Circuit(qf.X(0), qf.Y(2))
    assert circ0 == qf.Circuit(qf.X(0), qf.Y(2), qubits=(0, 2))
    assert circ0 != qf.Circuit(qf.CNot(0, 2))
    assert circ0 != qf.Circuit(qf.X(0), qf.Y(1))
    assert circ0 != qf.Circuit(qf.X(0), qf.X(2))
    assert circ0 != "NOT A CICUIT"
    assert circ0 != qf.Moment(qf.X(0), qf.Y(2))
    assert circ0 != qf.Circuit(qf.X(0), qf.Y(2), qubits=(2, 0))


def test_Circuit_repr() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Y(2), qf.Z(1))
    circ1 = eval(repr(circ0), qf.__dict__)
    assert circ0 == circ1

    circ0 = qf.Circuit(qf.X(0), qf.Y(2), qf.Z(1), qubits=(0, 2, 1))
    circ1 = eval(repr(circ0), qf.__dict__)
    assert circ0 == circ1
