# Copyright 2020-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest

import quantumflow as qf
from quantumflow.transpile import _guess_format, transpile, TRANSPILE_FORMATS

# Check which optional dependencies are available
pyquil = pytest.importorskip("pyquil", reason="pyquil not installed")
braket = pytest.importorskip("braket", reason="braket not installed")
cirq = pytest.importorskip("cirq", reason="cirq not installed")
qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
qutip_qip = pytest.importorskip("qutip_qip", reason="qutip-qip not installed")

# qsimcirq is optional - some tests will skip if not available
try:
    import qsimcirq
    HAS_QSIM = True
except ImportError:
    HAS_QSIM = False

from quantumflow import xbraket, xcirq, xforest, xqiskit, xqutip


def test_transpile_formats_constant() -> None:
    """Test that TRANSPILE_FORMATS contains expected formats."""
    assert "quantumflow" in TRANSPILE_FORMATS
    assert "cirq" in TRANSPILE_FORMATS
    assert "qiskit" in TRANSPILE_FORMATS
    assert "braket" in TRANSPILE_FORMATS
    assert "pyquil" in TRANSPILE_FORMATS
    assert "qasm" in TRANSPILE_FORMATS
    assert "quirk" in TRANSPILE_FORMATS
    assert "qsim" in TRANSPILE_FORMATS
    assert "qutip" in TRANSPILE_FORMATS


def test_guess_format_quantumflow() -> None:
    """Test format detection for QuantumFlow circuits."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    assert _guess_format(circ) == "quantumflow"


def test_guess_format_cirq() -> None:
    """Test format detection for Cirq circuits."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c0 = xcirq.circuit_to_cirq(circ)
    assert _guess_format(c0) == "cirq"


def test_guess_format_braket() -> None:
    """Test format detection for Braket circuits."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c1 = xbraket.circuit_to_braket(circ)
    assert _guess_format(c1) == "braket"


def test_guess_format_pyquil() -> None:
    """Test format detection for PyQuil programs."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c2 = xforest.circuit_to_pyquil(circ)
    assert _guess_format(c2) == "pyquil"


def test_guess_format_qiskit() -> None:
    """Test format detection for Qiskit circuits."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c3 = xqiskit.circuit_to_qiskit(circ)
    assert _guess_format(c3) == "qiskit"


def test_guess_format_qasm() -> None:
    """Test format detection for QASM strings."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c4 = xqiskit.circuit_to_qasm(circ)
    assert _guess_format(c4) == "qasm"


def test_guess_format_qutip() -> None:
    """Test format detection for QuTiP circuits."""
    circ = qf.Circuit(qf.X(0), qf.Z(1))
    c5 = xqutip.circuit_to_qutip(circ)
    assert _guess_format(c5) == "qutip"


def test_guess_format_unknown() -> None:
    """Test that unknown formats raise ValueError."""
    with pytest.raises(ValueError, match="Unknown source format"):
        _guess_format(12345)

    with pytest.raises(ValueError, match="Unknown source format"):
        _guess_format({"not": "a circuit"})


# Formats that don't require qsim
circuit_formats_no_qsim = [
    "quantumflow",
    "cirq",
    "braket",
    "pyquil",
    "qiskit",
    "qasm",
    "qutip",
]


@pytest.mark.parametrize("circuit_format", circuit_formats_no_qsim)
def test_transpile_roundtrip(circuit_format: str) -> None:
    """Test roundtrip transpilation for each format."""
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))

    circ1 = transpile(circ0, output_format=circuit_format)
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


@pytest.mark.parametrize("circuit_format", circuit_formats_no_qsim)
def test_transpile_with_translation(circuit_format: str) -> None:
    """Test transpilation with gate translation for complex gates."""
    circ0 = qf.Circuit(
        qf.X(0), qf.Z(1), qf.Margolus(0, 1, 2), qf.Can(0.1, 0.2, 0.3, 2, 3)
    )

    circ1 = transpile(circ0, output_format=circuit_format)
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


@pytest.mark.skipif(not HAS_QSIM, reason="qsimcirq not installed")
def test_transpile_qsim() -> None:
    """Test transpilation to qsim format."""
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))
    circ1 = transpile(circ0, output_format="qsim")
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


@pytest.mark.skipif(not HAS_QSIM, reason="qsimcirq not installed")
def test_transpile_qsim_with_translation() -> None:
    """Test transpilation to qsim with complex gates."""
    circ0 = qf.Circuit(
        qf.X(0), qf.Z(1), qf.Margolus(0, 1, 2), qf.Can(0.1, 0.2, 0.3, 2, 3)
    )
    circ1 = transpile(circ0, output_format="qsim")
    circ2 = transpile(circ1, output_format="quantumflow")
    assert qf.circuits_close(circ0, circ2)


def test_transpile_quirk() -> None:
    """Test transpilation to Quirk format (output only)."""
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))
    result = transpile(circ0, output_format="quirk")
    # Quirk output is a JSON string
    assert isinstance(result, str)
    assert "cols" in result  # Quirk JSON has 'cols' key


def test_transpile_across_formats() -> None:
    """Transpile to each supported format in turn, then back to QF."""
    circ0 = qf.Circuit(qf.Margolus(0, 1, 2))

    circ1 = circ0
    for f0 in circuit_formats_no_qsim:
        circ1 = transpile(circ0, output_format=f0)

    circ2 = transpile(circ1)
    assert qf.circuits_close(circ0, circ2)


def test_transpile_unknown_input() -> None:
    """Test that unknown input types raise ValueError."""
    with pytest.raises(ValueError, match="Unknown source format"):
        transpile(19939848)


def test_transpile_unknown_output_format() -> None:
    """Test that unknown output format raises ValueError."""
    circ0 = qf.Circuit(qf.Margolus(0, 1, 2))
    with pytest.raises(ValueError, match="Unknown output format"):
        transpile(circ0, output_format="NOT_A_FORMAT")


def test_transpile_empty_circuit() -> None:
    """Test transpilation of empty circuits."""
    circ0 = qf.Circuit()
    # Note: qutip doesn't handle empty circuits (max() on empty qubits fails)
    formats_for_empty = [f for f in circuit_formats_no_qsim if f != "qutip"]
    for fmt in formats_for_empty:
        circ1 = transpile(circ0, output_format=fmt)
        circ2 = transpile(circ1, output_format="quantumflow")
        assert len(circ2) == 0


def test_transpile_single_qubit() -> None:
    """Test transpilation of single-qubit circuits."""
    circ0 = qf.Circuit(qf.H(0), qf.T(0), qf.S(0))
    for fmt in circuit_formats_no_qsim:
        circ1 = transpile(circ0, output_format=fmt)
        circ2 = transpile(circ1, output_format="quantumflow")
        assert qf.circuits_close(circ0, circ2)


def test_transpile_default_output() -> None:
    """Test that default output format is quantumflow."""
    circ0 = qf.Circuit(qf.X(0), qf.Z(1))
    qk_circ = xqiskit.circuit_to_qiskit(circ0)

    # transpile with default should return quantumflow Circuit
    result = transpile(qk_circ)
    assert isinstance(result, qf.Circuit)
    assert qf.circuits_close(circ0, result)
