# Copyright 2020-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from typing import Any

from .circuits import Circuit

__all__ = "transpile", "TRANSPILE_FORMATS"

TRANSPILE_FORMATS = "qasm", "cirq", "braket", "pyquil", "qiskit", "quirk", "quantumflow"


def transpile(
    circuit: Any, output_format: str = "quantumflow", literal: bool = False
) -> Any:
    """Transpile a quantum circuit between different quantum libraries.

    Supported Formats:
        braket:
            An Amazon braket Circuit
        cirq:
            A Google Cirq Circuit
        pyquil:
            A Rigetti PyQuil Program.
        qasm:
            A QASM circuit, as a string.
        qiskit:
            An IBM qiskit QuantumCircuit.
        quantumflow:
            A native QuantumFlow Circuit.
        quirk:
            A quirk JSON formatted string. (Currently only supported for output)


    Args:
        circuit: A quantum circuit. The input format is inferred from the type
            and value of the circuit.
        output_format:
            The desired format for the output. One of "qasm", "cirq", "braket",
            "pyquil", "qiskit", "quirk", or "quantumflow"
        literal:
            Require a literal gate-for-gate translation (Default False).
            Different libraries support different sets of quantum gates. Normally
            we automatically translate unsupported gates into equivalent sequences of
            supported gates. If a literal translation is requested, encountering
            unsupported gates will instead raise an exception.

    Returns:
        An equivalent quantum circuit transpiled to a new format.

    Raises:
        ValueError: On unknown 'output_format' string.
        ModuleNotFoundError: If required external package isn't installed.
        NotImplementedError: When a literal translation encounters a gate that is not
            supported by the output format.
    """

    intermediate_form = _transpile_from(circuit)
    final_form = _transpile_to(intermediate_form, output_format, literal)

    return final_form


def _transpile_from(circuit: Any) -> Circuit:

    input_format = _guess_format(circuit)

    if input_format == "quantumflow":
        return circuit

    if input_format == "cirq":
        from . import xcirq

        return xcirq.cirq_to_circuit(circuit)

    if input_format == "braket":
        from . import xbraket

        return xbraket.braket_to_circuit(circuit)

    if input_format == "pyquil":
        from . import xforest

        return xforest.pyquil_to_circuit(circuit)

    if input_format == "qiskit":
        from . import xqiskit

        return xqiskit.qiskit_to_circuit(circuit)

    if input_format == "qasm":
        from . import xqiskit

        return xqiskit.qasm_to_circuit(circuit)

    raise ValueError(f"Unknown input format: {input_format}")  # pragma: no cover


def _guess_format(circuit: Any) -> str:
    typestr = str(type(circuit))

    if isinstance(circuit, Circuit):
        return "quantumflow"

    if "cirq" in typestr and "Circuit" in typestr:
        return "cirq"

    if "braket" in typestr and "Circuit" in typestr:
        return "braket"

    if "pyquil" in typestr and "Program" in typestr:
        return "pyquil"

    if "qiskit" in typestr and "QuantumCircuit" in typestr:
        return "qiskit"

    if isinstance(circuit, str) and "OPENQASM" in circuit:
        return "qasm"

    raise ValueError(f"Unknown source format for circuit: {typestr}")


def _transpile_to(circuit: Circuit, output_format: str, literal: bool) -> Any:
    translate = not literal

    if output_format == "quantumflow":
        return circuit

    if output_format == "cirq":
        from . import xcirq

        return xcirq.circuit_to_cirq(circuit, translate)

    if output_format == "braket":
        from . import xbraket

        return xbraket.circuit_to_braket(circuit, translate)

    if output_format == "pyquil":
        from . import xforest

        return xforest.circuit_to_pyquil(circuit, translate)

    if output_format == "qiskit":
        from . import xqiskit

        return xqiskit.circuit_to_qiskit(circuit, translate)

    if output_format == "qasm":
        from . import xqiskit

        return xqiskit.circuit_to_qasm(circuit, translate)

    if output_format == "quirk":
        from . import xquirk

        return xquirk.circuit_to_quirk(circuit, translate=True)

    raise ValueError(f"Unknown output format: {output_format}")
