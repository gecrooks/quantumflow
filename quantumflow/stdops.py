# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
========================
Miscellaneous Operations
========================


.. currentmodule:: quantumflow

Various standard operations on quantum states, which aren't gates,
channels, circuits, or DAGCircuit's.

.. autoclass :: Moment

.. autoclass :: Measure
.. autoclass :: Reset
.. autoclass :: Initialize
.. autoclass :: Barrier
.. autoclass :: Store
.. autoclass :: If
.. autoclass :: Display
.. autoclass :: StateDisplay
.. autoclass :: ProbabilityDisplay
.. autoclass :: DensityDisplay
.. autoclass :: Projection

"""

import textwrap
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import tensors, utils
from .circuits import Circuit
from .config import CIRCUIT_INDENT
from .gates import P0, P1
from .ops import Channel, Gate, Operation, Unitary
from .qubits import Qubit, Qubits
from .states import Density, State
from .tensors import QubitTensor
from .var import Variable

__all__ = [
    "Moment",
    "Measure",
    "Reset",
    "Initialize",
    "Barrier",
    "Store",
    "If",
    "Display",
    "StateDisplay",
    "ProbabilityDisplay",
    "DensityDisplay",
    "Projection",
]


class Moment(Sequence, Operation):
    """
    Represents a collection of Operations that operate on disjoint qubits,
    so that they may be applied at the same moment of time.
    """

    def __init__(self, *elements: Union[Iterable[Operation], Operation]) -> None:
        if len(elements) == 1 and isinstance(elements[0], Iterable):
            elements = elements[0]  # type: ignore

        circ = Circuit(Circuit(elements).flat())  # type: ignore
        qbs = list(q for elem in circ for q in elem.qubits)
        if len(qbs) != len(set(qbs)):
            raise ValueError("Qubits of operations within Moments " "must be disjoint.")

        super().__init__(qubits=qbs)
        self._circ = circ

    def __getitem__(self, key: Union[int, slice]) -> Any:
        return self._circ[key]

    def __len__(self) -> int:
        return self._circ.__len__()

    def __iter__(self) -> Iterator[Operation]:
        yield from self._circ

    def run(self, ket: State = None) -> State:
        return self._circ.run(ket)

    def evolve(self, rho: Density = None) -> Density:
        return self._circ.evolve(rho)

    def asgate(self) -> "Gate":
        return self._circ.asgate()

    def aschannel(self) -> "Channel":
        return self._circ.aschannel()

    @property
    def H(self) -> "Moment":
        return Moment(self._circ.H)

    def __str__(self) -> str:
        circ_str = "\n".join([str(elem) for elem in self])
        circ_str = textwrap.indent(circ_str, " " * CIRCUIT_INDENT)
        return "\n".join([self.name, circ_str])

    def on(self, *qubits: Qubit) -> "Moment":
        return Moment(Circuit(self).on(*qubits))

    def rewire(self, labels: Dict[Qubit, Qubit]) -> "Moment":
        return Moment(Circuit(self).rewire(labels))

    @property
    def params(self) -> Tuple[Variable, ...]:
        return tuple(item for elem in self for item in elem.params)

    def param(self, name: str) -> Variable:
        raise ValueError("Cannot lookup parameters by name for composite operations")


# end class Moment


class Measure(Operation):
    """Measure a quantum bit and copy result to a classical bit"""

    _diagram_labels = ["M({cbit})"]

    def __init__(self, qubit: Qubit, cbit: Hashable = None) -> None:
        if cbit is None:
            cbit = qubit

        super().__init__(qubits=[qubit])
        self.qubit = qubit
        self.cbit = cbit

    def __str__(self) -> str:
        if self.cbit != self.qubit:
            return f"{self.name} {self.qubit} {self.cbit}"
        return f"{self.name} {self.qubit}"

    def run(self, ket: State) -> State:
        prob_zero = P0(self.qubit).run(ket).norm()

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            ket = P0(self.qubit).run(ket).normalize()
            ket = ket.store({self.cbit: 0})
        else:  # measure one
            ket = P1(self.qubit).run(ket).normalize()
            ket = ket.store({self.cbit: 1})
        return ket

    def evolve(self, rho: Density) -> Density:
        p0 = P0(self.qubit).aschannel()
        p1 = P1(self.qubit).aschannel()

        prob_zero = p0.evolve(rho).norm()

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            rho = p0.evolve(rho).normalize()
            rho = rho.store({self.cbit: 0})
        else:  # measure one
            rho = p1.evolve(rho).normalize()
            rho = rho.store({self.cbit: 1})
        return rho


# FIXME: Can't have zero qubits
# Having no qubits specified screws up visualization
# and dagc
class Reset(Operation):
    r"""An operation that resets qubits to zero irrespective of the
    initial state.
    """

    _diagram_labels = ["┤ ⟨0|"]
    _diagram_noline = True

    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = ()
        super().__init__(qubits)

        self._gate = Unitary(tensor=[[1, 1], [0, 0]], qubits=[0])

    def run(self, ket: State) -> State:
        if self.qubits:
            qubits = self.qubits
        else:
            qubits = ket.qubits

        for q in qubits:
            gate = self._gate.on(q)
            ket = gate.run(ket)
        ket = ket.normalize()
        return ket

    def evolve(self, rho: Density) -> Density:
        # TODO
        raise TypeError("Not yet implemented")

    def asgate(self) -> Gate:
        raise TypeError("Reset not convertible to Gate")

    # FIXME?
    def aschannel(self) -> Channel:
        raise TypeError("Reset not convertible to Channel")

    def __str__(self) -> str:
        if self.qubits:
            return "Reset " + " ".join([str(q) for q in self.qubits])
        return "Reset"


class Initialize(Operation):
    """ An operation that initializes the quantum state"""

    def __init__(self, ket: State):
        self._ket = ket
        self._qubits = ket.qubits  # FIXME
        super().__init__(ket.qubits)

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return self._ket.tensor

    def run(self, ket: State) -> State:
        return self._ket.permute(ket.qubits)

    def evolve(self, rho: Density) -> Density:
        return self._ket.permute(rho.qubits).asdensity()

    # TODO: aschannel? __str___?


# TODO: Could be a Gate
# FIXME: Interface
class Barrier(Operation):
    """An operation that prevents reordering of operations across the barrier.
    Has no effect on the quantum state."""

    interchangable = True
    _diagram_labels = ["┼"]
    _diagram_noline = True

    def __init__(self, *qubits: Qubit) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> "Barrier":
        return self  # Hermitian

    def run(self, ket: State) -> State:
        return ket  # NOP

    def evolve(self, rho: Density) -> Density:
        return rho  # NOP

    def __str__(self) -> str:
        return self.name + " " + " ".join(str(q) for q in self.qubits)


# FIXME: Does not work as written?
class Projection(Operation):
    """A projection operator, represented as a sequence of state vectors"""

    # TODO: evolve(), asgate(), aschannel()

    def __init__(self, states: Sequence[State]):
        self.states = states

        qbs = [q for state in self.states for q in state.qubits]  # gather
        qbs = list(set(qbs))  # unique
        qbs = sorted(qbs)  # sort

        super().__init__(qbs)

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        tensor = sum(
            state.tensor * tensors.inner(state.tensor, ket.tensor)
            for state in self.states
        )
        return State(tensor, qubits=ket.qubits)

    @property
    def H(self) -> "Projection":
        return self  # pragma: no cover  # TESTME


# end class Projection


class Store(Operation):
    """Store a value in the classical memory of the state."""

    def __init__(self, key: Hashable, value: Any, qubits: Qubits = ()) -> None:
        super().__init__(qubits=qubits)
        self.key = key
        self.value = value

    def run(self, ket: State) -> State:
        return ket.store({self.key: self.value})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.value})


# end class Store


class If(Operation):
    """
    Look up key in classical memory, and apply the given
    quantum operation only if the truth value matches.
    """

    def __init__(self, elem: Operation, key: Hashable, value: bool = True) -> None:
        super().__init__(qubits=elem.qubits)
        self.element = elem
        self.key = key
        self.value = value

    def run(self, ket: State) -> State:
        if ket.memory[self.key] == self.value:
            ket = self.element.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        if rho.memory[self.key] == self.value:
            rho = self.element.evolve(rho)
        return rho


# end class If


class Display(Operation):
    """A Display is an operation that extracts information from the
    quantum state and stores it in classical memory, without performing
    any effect on the qubits.
    """

    # Terminology 'Display' used by Quirk (https://algassert.com/quirk)
    # and cirq (cirq/ops/display.py).
    def __init__(self, key: Hashable, action: Callable, qubits: Qubits = ()) -> None:
        super().__init__(qubits=qubits)
        self.key = key
        self.action = action

    def run(self, ket: State) -> State:
        return ket.store({self.key: self.action(ket)})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.action(rho)})


# end class Display


class StateDisplay(Display):
    """
    Store a copy of the state in the classical memory. (This operation
    can be memory intensive, since it stores the entire quantum state.)
    """

    def __init__(self, key: Hashable, qubits: Qubits = ()) -> None:
        super().__init__(key, lambda x: x, qubits=qubits)


# TODO: Act on qubit subspace
class ProbabilityDisplay(Display):
    """
    Store the state probabilities in classical memory.
    """

    def __init__(self, key: Hashable, qubits: Qubits = ()) -> None:
        super().__init__(key, lambda state: state.probabilities(), qubits=qubits)


# TESTME
class DensityDisplay(Display):
    """
    Store the density matrix of given qubits in classical memory.
    """

    def __init__(self, key: Hashable, qubits: Qubits) -> None:
        super().__init__(key, lambda state: state.asdensity(qubits), qubits=qubits)


# fin
