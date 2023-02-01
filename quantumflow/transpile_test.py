# Copyright 2020-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest

import quantumflow as qf
from quantumflow import xbraket, xcirq, xforest, xqiskit, xqutip
from quantumflow.transpile import _guess_format, transpile

pytest.importorskip("pyquil")  # noqa: 402
pytest.importorskip("qsimcirq")  # noqa: 402
pytest.importorskip("braket")  # noqa: 402
pytest.importorskip("cirq")  # noqa: 402
pytest.importorskip("qiskit")  # noqa: 402


def test_guess_format() -> None:
    circ = qf.Circuit(qf.X(0), qf.Z(1))

    c0 = xcirq.circuit_to_cirq(circ)
    f0 = _guess_format(c0)
    assert f0 == "cirq"

    c1 = xbraket.circuit_to_braket(circ)
    f1 = _guess_format(c1)
    assert f1 == "braket"

    c2 = xforest.circuit_to_pyquil(circ)
    f2 = _guess_format(c2)
    assert f2 == "pyquil"

    c3 = xqiskit.circuit_to_qiskit(circ)
    f3 = _guess_format(c3)
    assert f3 == "qiskit"

    c4 = xqiskit.circuit_to_qasm(circ)
    f4 = _guess_format(c4)
    assert f4 == "qasm"

    c4 = xqutip.circuit_to_qutip(circ)
    f4 = _guess_format(c4)
    assert f4 == "qutip"


circuit_formats = [
    "quantumflow",
    "cirq",
    "braket",
    "pyquil",
    "qiskit",
    "qasm",
    "qsim",
    "qutip",
]


@pytest.mark.parametrize("circuit_format", circuit_formats)
def test_transpile(circuit_format: str) -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))

    circ1 = transpile(circ0, output_format=circuit_format)
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


@pytest.mark.parametrize("circuit_format", circuit_formats)
def test_transpile_translate(circuit_format: str) -> None:
    circ0 = qf.Circuit(
        qf.X(0), qf.Z(1), qf.Margolus(0, 1, 2), qf.Can(0.1, 0.2, 0.3, 2, 3)
    )

    circ1 = transpile(circ0, output_format=circuit_format)
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


def test_transpile_quirk() -> None:
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))
    _ = transpile(circ0, output_format="quirk")


def test_transpile_accross() -> None:
    """Transpile to each supported format in turn, then back to QF"""

    circ0 = qf.Circuit(qf.Margolus(0, 1, 2))

    circ1 = circ0
    for f0 in circuit_formats:
        circ1 = transpile(circ0, output_format=f0)

    circ2 = transpile(circ1)
    assert qf.circuits_close(circ0, circ2)
    print(circ2)


def test_transpile_errors() -> None:
    with pytest.raises(ValueError):
        _ = transpile(19939848)

    circ0 = qf.Circuit(qf.Margolus(0, 1, 2))
    with pytest.raises(ValueError):
        _ = transpile(circ0, output_format="NOT_A_FORMAT")
