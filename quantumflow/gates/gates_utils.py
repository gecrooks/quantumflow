
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import TextIO, Union
from functools import reduce
import numpy as np
import scipy

from .. import backend as bk
from ..qubits import Qubit, Qubits, qubits_count_tuple
from ..qubits import outer_product
from ..ops import Gate, Unitary
from .. import utils
from .gates_one import IDEN

from ..paulialgebra import Pauli

__all__ = ['identity_gate',
           'random_gate',
           'join_gates',
           'control_gate',
           'conditional_gate',
           'P0', 'P1',
           'unitary_from_hamiltonian',
           'print_gate',
           ]


# TODO: Not needed?
def identity_gate(qubits: Union[int, Qubits]) -> Gate:
    """Returns the K-qubit identity gate"""
    _, qubits = qubits_count_tuple(qubits)
    return IDEN(*qubits)


# TODO: Can also use circuit.asgate()
def join_gates(*gates: Gate) -> Unitary:
    """Direct product of two gates. Qubit count is the sum of each gate's
    bit count."""
    vectors = [gate.vec for gate in gates]
    vec = reduce(outer_product, vectors)
    return Unitary(vec.tensor, *vec.qubits)


def control_gate(control: Qubit, gate: Gate) -> Unitary:
    """Return a controlled unitary gate. Given a gate acting on K qubits,
    return a new gate on K+1 qubits prepended with a control bit. """

    if control in gate.qubits:
        raise ValueError('Gate and control qubits overlap')

    qubits = [control, *gate.qubits]
    gate_tensor = join_gates(P0(control), identity_gate(gate.qubits)).tensor \
        + join_gates(P1(control), gate).tensor
    controlled_gate = Unitary(gate_tensor, *qubits)

    return controlled_gate


def conditional_gate(control: Qubit, gate0: Gate, gate1: Gate) -> Unitary:
    """Return a conditional unitary gate. Do gate0 on bit 1 if bit 0 is zero,
    else do gate1 on 1"""
    assert gate0.qubits == gate1.qubits  # FIXME

    tensor = join_gates(P0(control), gate0).tensor
    tensor += join_gates(P1(control), gate1).tensor
    gate = Unitary(tensor, *[control, *gate0.qubits])
    return gate


def print_gate(gate: Gate, ndigits: int = 2,
               file: TextIO = None) -> None:
    """Pretty print a gate tensor

    Args:
        gate:
        ndigits:
        file: Stream to which to write. Defaults to stdout
    """
    N = gate.qubit_nb
    gate_tensor = gate.vec.asarray()
    lines = []
    for index, amplitude in np.ndenumerate(gate_tensor):
        ket = "".join([str(n) for n in index[0:N]])
        bra = "".join([str(index[n]) for n in range(N, 2*N)])
        if round(abs(amplitude)**2, ndigits) > 0.0:
            lines.append(f'{bra} -> {ket} : {amplitude}')
    lines.sort(key=lambda x: int(x[0:N]))
    print('\n'.join(lines), file=file)


# FIXME: P0 and P1 should be channels

class P0(Gate):
    r"""Project qubit to zero.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 0.
    """
    text_labels = ['|0><0|']

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct([[1, 0], [0, 0]])


class P1(Gate):
    r"""Project qubit to one.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 1.
    """
    text_labels = ['|1><1|']

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct([[0, 0], [0, 1]])


# FIXME: Change interface to *qubits
def random_gate(qubits: Union[int, Qubits]) -> Unitary:
    r"""Returns a random unitary gate on K qubits.

    Ref:
        "How to generate random matrices from the classical compact groups"
        Francesco Mezzadri, math-ph/0609050
    """
    N, qubits = qubits_count_tuple(qubits)
    unitary = utils.unitary_ensemble(2**N)
    return Unitary(unitary, *qubits, name=f'RAND{N}')


def unitary_from_hamiltonian(
        hamiltonian: Pauli,
        *qubits: Qubit,
        name: str = None) -> Unitary:
    """Create a Unitary gate U from a Pauli operator H, U = exp(-i H)"""
    # Note: Can't be a classmethod constructor on Unitary due to circular
    # imports.
    op = hamiltonian.asoperator(qubits)
    U = scipy.linalg.expm(-1j * op)
    return Unitary(U, *qubits, name=name)
