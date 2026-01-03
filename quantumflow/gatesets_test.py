# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.gatesets
"""

import quantumflow as qf
from quantumflow.gatesets import (
    BRAKET_GATES,
    CIRQ_GATES,
    LATEX_OPERATIONS,
    QISKIT_GATES,
    QSIM_GATES,
    QUIL_GATES,
    QUIRK_GATES,
    QUTIP_GATES,
    TERMINAL_GATES,
)
from quantumflow.ops import Gate, Operation


def test_gatesets_nonempty() -> None:
    """Verify all gate sets are non-empty."""
    assert len(BRAKET_GATES) > 0
    assert len(CIRQ_GATES) > 0
    assert len(LATEX_OPERATIONS) > 0
    assert len(QISKIT_GATES) > 0
    assert len(QSIM_GATES) > 0
    assert len(QUIL_GATES) > 0
    assert len(QUIRK_GATES) > 0
    assert len(QUTIP_GATES) > 0
    assert len(TERMINAL_GATES) > 0


def test_gatesets_contain_gate_types() -> None:
    """Verify all gate sets contain only Gate subclasses."""
    for gate_type in BRAKET_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in CIRQ_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in QISKIT_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in QSIM_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in QUIL_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in QUIRK_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in QUTIP_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"

    for gate_type in TERMINAL_GATES:
        assert issubclass(gate_type, Gate), f"{gate_type} is not a Gate subclass"


def test_latex_operations_contain_operation_types() -> None:
    """Verify LATEX_OPERATIONS contains only Operation subclasses."""
    for op_type in LATEX_OPERATIONS:
        assert issubclass(op_type, Operation), f"{op_type} is not an Operation subclass"


def test_gatesets_subset_relationships() -> None:
    """Verify expected subset relationships between gate sets."""
    # QSIM gates should be a subset of CIRQ gates
    assert QSIM_GATES.issubset(CIRQ_GATES), (
        f"QSIM_GATES has gates not in CIRQ_GATES: {QSIM_GATES - CIRQ_GATES}"
    )


def test_gatesets_common_gates() -> None:
    """Verify common gates are present in all gate sets."""
    # These gates should be in all hardware-targeting gate sets
    common_gates = {qf.I, qf.X, qf.Y, qf.Z, qf.H, qf.CNot}

    for gate_type in common_gates:
        assert gate_type in BRAKET_GATES, f"{gate_type} missing from BRAKET_GATES"
        assert gate_type in CIRQ_GATES, f"{gate_type} missing from CIRQ_GATES"
        assert gate_type in QISKIT_GATES, f"{gate_type} missing from QISKIT_GATES"
        assert gate_type in QUIL_GATES, f"{gate_type} missing from QUIL_GATES"
        assert gate_type in QUIRK_GATES, f"{gate_type} missing from QUIRK_GATES"
        assert gate_type in QUTIP_GATES, f"{gate_type} missing from QUTIP_GATES"


def test_terminal_gates_are_fundamental() -> None:
    """Verify terminal gates are appropriate for gate decomposition targets."""
    # Terminal gates should include basic single-qubit gates
    assert qf.I in TERMINAL_GATES
    assert qf.X in TERMINAL_GATES
    assert qf.Y in TERMINAL_GATES
    assert qf.Z in TERMINAL_GATES
    assert qf.H in TERMINAL_GATES

    # And CNot for two-qubit entanglement
    assert qf.CNot in TERMINAL_GATES


def test_gatesets_are_sets() -> None:
    """Verify gate sets are actually set objects (no duplicates)."""
    # Sets automatically remove duplicates, so we just verify the type
    assert isinstance(BRAKET_GATES, (set, frozenset))
    assert isinstance(CIRQ_GATES, (set, frozenset))
    assert isinstance(LATEX_OPERATIONS, (set, frozenset))
    assert isinstance(QISKIT_GATES, (set, frozenset))
    assert isinstance(QSIM_GATES, (set, frozenset))
    assert isinstance(QUIL_GATES, (set, frozenset))
    assert isinstance(QUIRK_GATES, (set, frozenset))
    assert isinstance(QUTIP_GATES, (set, frozenset))
    assert isinstance(TERMINAL_GATES, (set, frozenset))


def test_exported_gatesets() -> None:
    """Verify __all__ exports match the actual gate sets."""
    from quantumflow import gatesets

    for name in gatesets.__all__:
        assert hasattr(gatesets, name), f"{name} in __all__ but not defined"


# fin
