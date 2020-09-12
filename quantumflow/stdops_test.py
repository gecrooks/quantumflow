# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import pytest

import quantumflow as qf


def test_moment() -> None:
    circ = qf.Circuit()
    circ += qf.X(0)
    circ += qf.Swap(1, 2)

    moment = qf.Moment(circ)

    assert moment.qubits == (0, 1, 2)
    assert moment.run()
    assert moment.evolve()
    assert isinstance(moment.H, qf.Moment)

    circ += qf.Y(0)
    with pytest.raises(ValueError):
        moment = qf.Moment(circ)

    assert moment.asgate()
    assert moment.aschannel()

    circ1 = qf.Circuit(moment)
    assert len(circ1) == 2

    assert isinstance(moment[1], qf.Swap)

    moment1 = moment.on("a", "b", "c")

    moment2 = moment1.rewire({"a": 0, "b": 1, "c": 2})
    assert str(moment) == str(moment2)


def test_moment_params() -> None:
    circ = qf.Circuit()
    circ += qf.X(0) ** 0.3
    circ += qf.Swap(1, 2)
    moment = qf.Moment(circ)

    assert len(moment.params) == 1
    assert moment.params == (0.3,)

    with pytest.raises(ValueError):
        _ = moment.param("theta")


def test_measure() -> None:
    prog = qf.Circuit()
    prog += qf.Measure(0, ("c", 0))
    prog += qf.X(0)
    prog += qf.Measure(0, ("c", 1))
    ket = prog.run()

    assert ket.qubits == (0,)
    assert ket.memory[("c", 0)] == 0
    assert ket.memory[("c", 1)] == 1

    assert str(qf.Measure(4)) == "Measure 4"
    assert str(qf.Measure(4, "c0")) == "Measure 4 c0"


def test_barrier() -> None:
    circ = qf.Circuit()
    circ += qf.Barrier(0, 1, 2)
    circ += qf.Barrier(0, 1, 2).H
    circ.run()
    circ.evolve()

    assert str(qf.Barrier(0, 1, 2)) == "Barrier 0 1 2"


def test_if() -> None:
    circ = qf.Circuit()
    c = ["c0", "c1"]
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1

    circ = qf.Circuit()
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 0
    assert circ.evolve().memory[c[0]] == 0

    circ = qf.Circuit()
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1


def test_display_state() -> None:
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.StateDisplay(key="state0")

    ket = circ.run()
    assert "state0" in ket.memory

    rho = circ.evolve()
    assert "state0" in rho.memory


def test_display_probabilities() -> None:
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.ProbabilityDisplay(key="prob")

    ket = circ.run()
    assert "prob" in ket.memory

    rho = circ.evolve()
    assert "prob" in rho.memory


def test_density_display() -> None:
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.DensityDisplay(key="bloch1", qubits=[1])

    ket = circ.run()
    assert "bloch1" in ket.memory
    assert ket.memory["bloch1"].qubits == (1,)

    rho = circ.evolve()
    assert "bloch1" in rho.memory
    assert rho.memory["bloch1"].qubits == (1,)


def test_initialize() -> None:
    circ = qf.Circuit()
    circ += qf.H(1)
    ket = qf.random_state([0, 1, 2])
    circ += qf.Initialize(ket)

    assert circ.qubits == (0, 1, 2)
    assert qf.states_close(circ.run(), ket)

    assert qf.densities_close(circ.evolve(), ket.asdensity())


def test_reset() -> None:
    reset = qf.Reset(0, 1, 2)
    ket = qf.random_state([0, 1, 2, 3])
    ket = reset.run(ket)
    assert ket.tensor[1, 1, 0, 0] == 0.0
    assert str(reset) == "Reset 0 1 2"

    reset = qf.Reset()
    ket = qf.random_state([0, 1, 2, 3])
    ket = reset.run(ket)
    assert qf.states_close(ket, qf.zero_state([0, 1, 2, 3]))
    assert str(reset) == "Reset"

    with pytest.raises(TypeError):
        reset.evolve(qf.random_density([0, 1, 3]))

    with pytest.raises(TypeError):
        reset.asgate()

    with pytest.raises(TypeError):
        reset.aschannel()


# fin
