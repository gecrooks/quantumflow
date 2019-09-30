
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow


Circuit objects
###############
.. autoclass:: Circuit
    :members:

.. autoclass:: DAGCircuit
    :members:


Standard circuits
#################

.. autofunction:: qft_circuit
.. autofunction:: reversal_circuit
.. autofunction:: control_circuit
.. autofunction:: zyz_circuit
.. autofunction:: phase_estimation_circuit
.. autofunction:: addition_circuit
.. autofunction:: ghz_circuit
.. autofunction:: graph_circuit
.. autofunction:: graph_circuit_params


Visualizations
##############

.. autofunction:: circuit_to_diagram
.. autofunction:: circuit_to_image
.. autofunction:: circuit_to_latex
.. autofunction:: latex_to_image
"""

from typing import Sequence, Iterator, Iterable, Dict, Type, Any, Union
from math import pi
from itertools import chain
from collections import defaultdict
from collections.abc import MutableSequence
import textwrap

import numpy as np
import networkx as nx
from sympy import Symbol

from .qubits import Qubit, Qubits
from .states import State, Density, zero_state
from .ops import Operation, Gate, Channel
from .gates import control_gate, identity_gate
from .gates import H, CPHASE, SWAP, CNOT, X, TX, TY, TZ, CCNOT, ZZ, CZ

__all__ = ['Circuit',
           'count_operations',
           'map_gate',
           'qft_circuit',
           'reversal_circuit',
           'control_circuit',
           'zyz_circuit',
           'phase_estimation_circuit',
           'addition_circuit',
           'ghz_circuit',
           'graph_circuit',
           'graph_circuit_params',
           'graph_state_circuit']


class Circuit(MutableSequence, Operation):
    """A quantum Circuit contains a sequences of circuit elements.
    These can be any quantum Operation, including other circuits.

    QuantumFlow's circuit can only contain Operations. They do not contain
    control flow of other classical computations (similar to pyquil's
    protoquil). For hybrid algorithms involving control flow and other
    classical processing use QuantumFlow's Program class.
    """
    def __init__(self, elements: Iterable[Operation] = None) -> None:
        if elements is None:
            elements = []
        # TODO: Make elements private
        self.elements = list(elements)

    # Methods for MutableSequence
    def __getitem__(self, key: Union[int, slice]) -> Any:
        return self.elements[key]

    def __delitem__(self, key: Union[int, slice]) -> None:
        del self.elements[key]

    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        self.elements[key] = value

    def __len__(self) -> int:
        return self.elements.__len__()

    def insert(self, idx: int, value: Any) -> None:
        self.elements.insert(idx, value)

    def extend(self, other: Iterable[Any]) -> None:
        """Append gates from circuit to the end of this circuit"""
        if other is self:
            # We can go into infinite regress otherwise.
            other = list(self.elements)
        self.elements.extend(other)

    def add(self, other: 'Circuit') -> 'Circuit':
        """Concatenate gates and return new circuit"""
        return Circuit(chain(self, other))

    def __add__(self, other: 'Circuit') -> 'Circuit':
        return self.add(other)

    def __iadd__(self, other: Iterable[Any]) -> 'Circuit':
        self.extend(other)
        return self

    def __iter__(self) -> Iterator[Operation]:
        return self.elements.__iter__()

    def flat(self) -> Iterator[Operation]:
        """Iterate over all elemenary elements of Circuit,
        recursively flattening composite elements such as
        sub-Circuits, DAGCircuits, and Moments"""
        for elem in self:
            if hasattr(elem, 'flat'):
                yield from elem.flat()
            else:
                yield from elem

    def size(self) -> int:
        """Return the number of operations in this circuit"""
        return len(self.elements)

    @property
    def qubits(self) -> Qubits:
        """Returns: Sorted list of qubits acted upon by this circuit

        Raises:
            TypeError: If qubits cannot be sorted into unique order.
        """
        qbs = [q for elem in self.elements for q in elem.qubits]    # gather
        qbs = list(set(qbs))                                        # unique
        qbs = sorted(qbs)                                           # sort
        return tuple(qbs)

    def run(self, ket: State = None) -> State:
        """
        Apply the action of this circuit upon a state.

        If no initial state provided an initial zero state will be created.
        """
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits=qubits)

        for elem in self.elements:
            ket = elem.run(ket)
        return ket

    # DOCME
    def evolve(self, rho: Density = None) -> Density:
        if rho is None:
            qubits = self.qubits
            rho = zero_state(qubits=qubits).asdensity()

        for elem in self.elements:
            rho = elem.evolve(rho)
        return rho

    # DOCME: What gets raised if we can't construct a gate?
    def asgate(self) -> Gate:
        """
        Return the action of this circuit as a gate (If possible)
        """
        gate = identity_gate(self.qubits)
        for elem in self.elements:
            gate = elem.asgate() @ gate
        return gate

    # TESTME
    # DOCME
    def aschannel(self) -> Channel:
        chan = identity_gate(self.qubits).aschannel()
        for elem in self.elements:
            chan = elem.aschannel() @ chan
        return chan

    @property
    def H(self) -> 'Circuit':
        """Returns the Hermitian conjugate of this circuit.
        If all the subsidiary gates are unitary, returns the circuit inverse.
        """
        return Circuit([elem.H for elem in self.elements[::-1]])

    # TESTME
    def __str__(self) -> str:
        circ_str = '\n'.join([str(elem) for elem in self])
        circ_str = textwrap.indent(circ_str, '    ')
        return '\n'.join([self.name, circ_str])

    # TESTME DOCME
    def resolve(self, resolver: Dict[Symbol, float]) -> 'Circuit':
        """Resolve symbolic parameters"""
        return Circuit(op.resolve(resolver) for op in self)

    # TODO: overide params, so that fails, or returns all paramerters?


# End class Circuit


def count_operations(elements: Iterable[Operation]) \
        -> Dict[Type[Operation], int]:
    """Return a count of different operation types given a collection of
    operations, such as a Circuit or DAGCircuit
    """
    op_count: Dict[Type[Operation], int] = defaultdict(int)
    for elem in elements:
        op_count[type(elem)] += 1
    return dict(op_count)


def map_gate(gate: Gate, args: Sequence[Qubits]) -> Circuit:
    """Applies the same gate to all input qubits in the argument list.

    >>> circ = qf.map_gate(qf.H(), [[0], [1], [2]])
    >>> print(circ)
    Circuit
        H(0)
        H(1)
        H(2)

    """
    circ = Circuit()

    for qubits in args:
        circ += gate.relabel(qubits)

    return circ


# TODO: Move standard circuits to stdcircuits module?

# FIXME: Use ZZ gates, not CPHASE
# TODO: Add circuit diagram
def qft_circuit(qubits: Qubits) -> Circuit:
    """Returns the Quantum Fourier Transform circuit"""
    # Kudos: Adapted from Rigetti Grove, grove/qft/fourier.py

    N = len(qubits)
    circ = Circuit()
    for n0 in range(N):
        q0 = qubits[n0]
        circ += H(q0)
        for n1 in range(n0+1, N):
            q1 = qubits[n1]
            angle = pi / 2 ** (n1-n0)
            circ += CPHASE(angle, q1, q0)
    circ.extend(reversal_circuit(qubits))
    return circ


def reversal_circuit(qubits: Qubits) -> Circuit:
    """Returns a circuit to reverse qubits"""
    N = len(qubits)
    circ = Circuit()
    for n in range(N // 2):
        circ += SWAP(qubits[n], qubits[N-1-n])
    return circ


def control_circuit(controls: Qubits, gate: Gate) -> Circuit:
    """
    Returns a circuit for a target gate controlled by
    a collection of control qubits. [Barenco1995]_

    Uses a number of gates quadratic in the number of control qubits.

    .. [Barenco1995] A. Barenco, C. Bennett, R. Cleve (1995) Elementary Gates
        for Quantum Computation`<https://arxiv.org/abs/quant-ph/9503016>`_
        Sec 7.2
    """
    # Kudos: Adapted from Rigetti Grove's utility_programs.py
    # grove/utils/utility_programs.py::ControlledProgramBuilder

    circ = Circuit()
    if len(controls) == 1:
        q0 = controls[0]
        if isinstance(gate, X):
            circ += CNOT(q0, gate.qubits[0])
        else:
            # FIXME: would be better to return circuit
            cgate = control_gate(q0, gate)
            circ += cgate
    else:
        circ += control_circuit(controls[-1:], gate ** 0.5)
        circ += control_circuit(controls[0:-1], X(controls[-1]))
        circ += control_circuit(controls[-1:], gate ** -0.5)
        circ += control_circuit(controls[0:-1], X(controls[-1]))
        circ += control_circuit(controls[0:-1], gate ** 0.5)
    return circ


def zyz_circuit(t0: float, t1: float, t2: float, q0: Qubit = 0) -> Circuit:
    """Circuit equivalent of 1-qubit ZYZ gate"""
    return euler_circuit(t0, t1, t2, q0, 'ZYZ')


def euler_circuit(t0: float, t1: float, t2: float,
                  q0: Qubit = 0,
                  euler: str = 'ZYZ',) -> Circuit:
    """
    DOCME

    The 'euler' argument can be used to specify any of the 6 Euler
    decompositions: 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ' (Default)
    """
    euler_circ = {
        'XYX': (TX, TY, TX),
        'XZX': (TX, TZ, TX),
        'YXY': (TY, TX, TY),
        'YZY': (TY, TZ, TY),
        'ZXZ': (TZ, TX, TZ),
        'ZYZ': (TZ, TY, TZ)
    }

    gate0, gate1, gate2 = euler_circ[euler]
    circ = Circuit()
    circ += gate0(t0, q0)
    circ += gate1(t1, q0)
    circ += gate2(t2, q0)
    return circ


def phase_estimation_circuit(gate: Gate, outputs: Qubits) -> Circuit:
    """Returns a circuit for quantum phase estimation.

    The gate has an eigenvector with eigenvalue e^(i 2 pi phase). To
    run the circuit, the eigenvector should be set on the gate qubits,
    and the output qubits should be in the zero state. After evolution and
    measurement, the output qubits will be (approximately) a binary fraction
    representation of the phase.

    The output registers can be converted with the aid of the
    quantumflow.utils.bitlist_to_int() method.

    >>> import numpy as np
    >>> import quantumflow as qf
    >>> N = 8
    >>> phase = 1/4
    >>> gate = qf.RZ(-4*np.pi*phase, N)
    >>> circ = qf.phase_estimation_circuit(gate, range(N))
    >>> res = circ.run().measure()[0:N]
    >>> est_phase = int(''.join([str(d) for d in res]), 2) / 2**N # To float
    >>> print(phase, est_phase)
    0.25 0.25

    """
    circ = Circuit()
    circ += map_gate(H(), list(zip(outputs)))  # Hadamard on all output qubits

    for cq in reversed(outputs):
        cgate = control_gate(cq, gate)
        circ += cgate
        gate = gate @ gate

    circ += qft_circuit(outputs).H

    return circ


def addition_circuit(
        addend0: Qubits,
        addend1: Qubits,
        carry: Qubits) -> Circuit:
    """Returns a quantum circuit for ripple-carry addition. [Cuccaro2004]_

    Requires two carry qubit (input and output). The result is returned in
    addend1.

    .. [Cuccaro2004]
        A new quantum ripple-carry addition circuit, Steven A. Cuccaro,
        Thomas G. Draper, Samuel A. Kutin, David Petrie Moulton
        arXiv:quant-ph/0410184 (2004)
    """

    if len(addend0) != len(addend1):
        raise ValueError('Number of addend qubits must be equal')

    if len(carry) != 2:
        raise ValueError('Expected 2 carry qubits')

    def _maj(qubits: Qubits) -> Circuit:
        q0, q1, q2 = qubits
        circ = Circuit()
        circ += CNOT(q2, q1)
        circ += CNOT(q2, q0)
        circ += CCNOT(q0, q1, q2)
        return circ

    def _uma(qubits: Qubits) -> Circuit:
        q0, q1, q2 = qubits
        circ = Circuit()
        circ += CCNOT(q0, q1, q2)
        circ += CNOT(q2, q0)
        circ += CNOT(q0, q1)
        return circ

    qubits = [carry[0]] + list(chain.from_iterable(
        zip(reversed(addend1), reversed(addend0)))) + [carry[1]]

    circ = Circuit()

    for n in range(0, len(qubits)-3, 2):
        circ += _maj(qubits[n:n+3])

    circ += CNOT(qubits[-2], qubits[-1])

    for n in reversed(range(0, len(qubits)-3, 2)):
        circ += _uma(qubits[n:n+3])

    return circ


def ghz_circuit(qubits: Qubits) -> Circuit:
    """Returns a circuit that prepares a multi-qubit Bell state from the zero
    state.
    """
    circ = Circuit()

    circ += H(qubits[0])
    for q0 in range(0, len(qubits)-1):
        circ += CNOT(qubits[q0], qubits[q0+1])

    return circ


def graph_circuit_params(
        topology: nx.Graph,
        steps: int,
        init_bias: float = 0.0,
        init_scale: float = 0.01) -> Sequence[float]:
    """Return a set of initial parameters for graph_circuit()"""
    N = len(topology.nodes())
    K = len(topology.edges())
    total = N+N*2*(steps+1) + K*steps
    params = np.random.normal(loc=init_bias, scale=init_scale,
                              size=[total])

    return params


def graph_circuit(
        topology: nx.Graph,
        steps: int,
        params: Sequence[float],
        ) -> Circuit:
    """
    Create a multilayer parameterized circuit given a graph of connected
    qubits.

    We alternate beween applying a sublayer of arbitary 1-qubit gates to all
    qubits, and a sublayer of ZZ gates between all connected qubits.

    In practive the pattern is TZ, TX, TZ, (ZZ, TX, TZ )*, where TZ are
    1-qubit Z rotations, and TX and 1-qubits X rotations. Since a Z rotation
    commutes accross a ZZ, we only need one Z sublayer per layer.

    Our fundametnal 2-qubit interaction is the Ising like ZZ gate. We could
    apply a more general gate, such as the universal Canonical gate. But ZZ
    gates commute with each other, whereas other choice of gate would not,
    which would necessitate specifing the order of all 2-qubit gates within
    the layer.
    """
    def tx_layer(topology: nx.Graph, layer_params: Sequence[float]) -> Circuit:
        circ = Circuit()
        for p, q0 in zip(layer_params, topology.nodes()):
            circ += TX(p, q0)
        return circ

    def tz_layer(topology: nx.Graph, layer_params: Sequence[float]) -> Circuit:
        circ = Circuit()
        for p, q0 in zip(layer_params, topology.nodes()):
            circ += TZ(p, q0)
        return circ

    def zz_layer(topology: nx.Graph, layer_params: Sequence[float]) -> Circuit:
        circ = Circuit()
        for p, (q0, q1) in zip(layer_params, topology.edges()):
            circ += ZZ(p, q0, q1)
        return circ

    N = len(topology.nodes())
    K = len(topology.edges())

    circ = Circuit()

    n = 0
    circ += tz_layer(topology, params[n:n+N])
    n += N
    circ += tx_layer(topology, params[n:n+N])
    n += N
    circ += tz_layer(topology, params[n:n+N])
    n += N

    for _ in range(steps):
        circ += zz_layer(topology, params[n:n+K])
        n += K
        circ += tx_layer(topology, params[n:n+N])
        n += N
        circ += tz_layer(topology, params[n:n+N])
        n += N

    return circ


def graph_state_circuit(topology: nx.Graph) -> Circuit:
    """
    Return a circuit to create a graph state, given a
    particular graph topology.

    Refs:
        - [Wikipedia: Graph State](https://en.wikipedia.org/wiki/Graph_state)
    """
    circ = Circuit()

    for q in topology.nodes:
        circ += H(q)

    for q0, q1 in topology.edges:
        circ += CZ(q0, q1)

    return circ


# Fin
