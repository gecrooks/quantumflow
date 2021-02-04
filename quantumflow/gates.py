# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import TextIO

import numpy as np
import scipy

from . import tensors, utils
from .ops import Gate, Unitary
from .paulialgebra import Pauli
from .qubits import Qubit, Qubits
from .tensors import QubitTensor

__all__ = (
    "RandomGate",
    "join_gates",
    "print_gate",
    "P0",
    "P1",
    "unitary_from_hamiltonian",
    "conditional_gate",
)


# DOCME: Explain cv_interchangeable
class RandomGate(Unitary):
    r"""Returns a random unitary gate on K qubits.

    Ref:
        "How to generate random matrices from the classical compact groups"
        Francesco Mezzadri, math-ph/0609050
    """
    cv_interchangeable = True

    def __init__(self, qubits: Qubits) -> None:
        qubits = tuple(qubits)
        tensor = utils.unitary_ensemble(2 ** len(qubits))
        super().__init__(tensor, qubits)


# end class RandomGate


# DOCME: Can also use circuit.asgate()
def join_gates(gate0: Gate, gate1: Gate) -> Unitary:
    """Direct product of gates. Qubit count is the sum of each gate's
    bit count."""
    tensor = tensors.outer(gate0.tensor, gate1.tensor, rank=2)
    return Unitary(tensor, tuple(gate0.qubits) + tuple(gate1.qubits))


def print_gate(gate: Gate, ndigits: int = 2, file: TextIO = None) -> None:
    """Pretty print a gate tensor

    Args:
        gate:
        ndigits:
        file: Stream to which to write. Defaults to stdout
    """
    N = gate.qubit_nb
    gate_tensor = gate.tensor
    lines = []
    for index, amplitude in np.ndenumerate(gate_tensor):
        ket = "".join([str(n) for n in index[0:N]])
        bra = "".join([str(index[n]) for n in range(N, 2 * N)])
        if round(abs(amplitude) ** 2, ndigits) > 0.0:
            lines.append(f"{bra} -> {ket} : {amplitude}")
    lines.sort(key=lambda x: int(x[0:N]))
    print("\n".join(lines), file=file)


# FIXME: P0 and P1 should be channels or Projectors ???


class P0(Gate):
    r"""Project qubit to zero.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 0.
    """
    text_labels = ["|0><0|"]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor([[1, 0], [0, 0]])


class P1(Gate):
    r"""Project qubit to one.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 1.
    """
    text_labels = ["|1><1|"]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor([[0, 0], [0, 1]])


def unitary_from_hamiltonian(hamiltonian: Pauli, qubits: Qubits) -> Unitary:
    """Create a Unitary gate U from a Pauli operator H, U = exp(-i H)"""
    # Note: Can't be a classmethod constructor on Unitary due to circular
    # imports. (???)
    op = hamiltonian.asoperator(qubits)
    U = scipy.linalg.expm(-1j * op)
    return Unitary(U, qubits)


# TODO: Replace with class
def conditional_gate(control: Qubit, gate0: Gate, gate1: Gate) -> Unitary:
    """Return a conditional unitary gate. Do gate0 on bit 1 if bit 0 is zero,
    else do gate1 on 1"""
    assert gate0.qubits == gate1.qubits  # FIXME

    tensor = join_gates(P0(control), gate0).tensor
    tensor += join_gates(P1(control), gate1).tensor
    gate = Unitary(tensor, (control,) + tuple(gate0.qubits))
    return gate


# fin
