# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: gates

Multi-qubit gates
#################

Larger unitary computational gates that can be broken up into standard gates.

Danger: These multi-qubit gates have a variable, and possible large, number of qubits.
Explicitly creating the gate tensor may consume huge amounts of memory. Beware.

.. autoclass:: UnitaryGate
    :members:

.. autoclass:: DiagonalGate
    :members:

.. autoclass:: IdentityGate
    :members:

.. autoclass:: PauliGate
    :members:

.. autoclass:: MultiSwapGate
    :members:

.. autoclass:: ReversalGate
    :members:

.. autoclass:: CircularShiftGate
    :members:

.. autoclass:: QFTGate
    :members:

.. autoclass:: InvQFTGate
    :members:

.. autoclass:: ControlGate
    :members:

.. autoclass:: MultiplexedGate
    :members:

.. autoclass:: ConditionalGate
    :members:

.. autoclass:: MultiplexedRzGate
    :members:

.. autoclass:: RandomGate
    :members:


.. autoclass:: CompositeGate
    :members:

"""

from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import networkx as nx
import numpy as np
import scipy

from . import tensors, utils, var
from .circuits import Circuit
from .future import cached_property
from .ops import Channel, Gate, Operation, UnitaryGate
from .paulialgebra import Pauli, sX, sY, sZ
from .qubits import Qubit, Qubits
from .states import Density, State
from .stdgates import CNot, I, Ry, Rz, Swap, XPow, YPow, ZPow
from .tensors import QubitTensor, asqutensor
from .var import Variable

__all__ = (
    "IdentityGate",
    "PauliGate",
    "MultiSwapGate",
    "ReversalGate",
    "CircularShiftGate",
    "QFTGate",
    "InvQFTGate",
    "DiagonalGate",
    "ControlGate",
    "MultiplexedGate",
    "ConditionalGate",
    "MultiplexedRzGate",
    "MultiplexedRyGate",
    "RandomGate",
    "CompositeGate",
)


class IdentityGate(Gate):
    r"""
    The multi-qubit identity gate.
    """

    cv_interchangeable = True
    cv_hermitian = True
    cv_tensor_structure = "identity"

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor(np.eye(2**self.qubit_nb))

    def __pow__(self, t: Variable) -> "IdentityGate":
        return self

    @property
    def H(self) -> "IdentityGate":
        return self

    def _diagram_labels_(self) -> List[str]:
        return ["I"]


# end class IdentityGate


class CompositeGate(Gate):
    """
    A quantum gate represented by a sequence of quantum gates.
    """

    def __init__(self, *elements: Gate, qubits: Optional[Qubits] = None) -> None:
        circ = Circuit(Circuit(elements).flat(), qubits=qubits)

        for elem in circ:
            if not isinstance(elem, Gate):
                raise ValueError("A CompositeGate must be composed of Gates")

        super().__init__(qubits=circ.qubits)
        self.circuit = circ

    def run(self, ket: Optional[State] = None) -> State:
        return self.circuit.run(ket)

    def evolve(self, rho: Optional[Density] = None) -> Density:
        return self.circuit.evolve(rho)

    def aschannel(self) -> "Channel":
        return self.circuit.aschannel()

    @cached_property
    def tensor(self) -> QubitTensor:
        return self.circuit.asgate().tensor

    @property
    def H(self) -> "CompositeGate":
        return CompositeGate(*self.circuit.H, qubits=self.qubits)

    def __str__(self) -> str:
        lines = str(self.circuit).split("\n")
        lines[0] = super().__str__()
        return "\n".join(lines)

    def on(self, *qubits: Qubit) -> "CompositeGate":
        return CompositeGate(*self.circuit.on(*qubits), qubits=qubits)

    def rewire(self, labels: Dict[Qubit, Qubit]) -> "CompositeGate":
        circ = self.circuit.rewire(labels)
        return CompositeGate(*circ, qubits=circ.qubits)

    @property
    def params(self) -> Tuple[Variable, ...]:
        return tuple(item for elem in self.circuit for item in elem.params)

    def param(self, name: str) -> Variable:
        raise ValueError("Cannot lookup parameters by name for composite operations")


# end class CompositeGate


# TODO: Decompose
class ControlGate(Gate):
    """A controlled unitary gate. Given C control qubits and a
    gate acting on K qubits, return a controlled gate with C+K qubits.
    The optional axes argument specifies the basis of the control
    qubits. The length of the sting should be the same as the number of control
    qubits. The default axis 'Z' is standard control in the standard 'z'
    (computational) basis. Anti-control, where the gate is activated with the zero
    state, is specified by 'z'
        ==== ====== ======  ============
        Axis Symbol Basis
        ==== ====== ====================
         X     ⊖    x-axis anti-control
         x     ⊕    x-basis control
         Y     ⊘    y-axis anti-control
         y     ⊗    y-basis control
         z     ○    z-basis anti-control
         Z     ●    z-axis control
        ==== ====== ====================
    The symbols are used in circuit diagrams. Z-basis control '●' and anti-control '○'
    symbols are standard, the rest are adapted from quirk.
    """

    # Note: ControlGate and StdCtrlGate share interface and code.
    # But unification probably not worth the trouble

    def __init__(
        self, target: Gate, controls: Qubits, axes: Optional[str] = None
    ) -> None:
        controls = tuple(controls)
        qubits = tuple(controls) + tuple(target.qubits)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Control and gate qubits overlap")

        if axes is None:
            axes = "Z" * len(controls)
        assert len(axes) == len(controls)

        super().__init__(qubits)

        self.target = target
        self.axes = axes

    @property
    def control_qubits(self) -> Qubits:
        return self.qubits[: -self.target.qubit_nb]

    @property
    def target_qubits(self) -> Qubits:
        return self.target.qubits

    @property
    def control_qubit_nb(self) -> int:
        return self.qubit_nb - self.target.qubit_nb

    @property
    def target_qubit_nb(self) -> int:
        return self.target.qubit_nb

    @property
    def hamiltonian(self) -> Pauli:
        ctrlham = {
            "X": (1 - sX(0)) / 2,
            "x": (1 + sX(0)) / 2,
            "Y": (1 - sY(0)) / 2,
            "y": (1 + sY(0)) / 2,
            "Z": (1 - sZ(0)) / 2,
            "z": (1 + sZ(0)) / 2,
        }

        ham = self.target.hamiltonian
        for q, axis in zip(self.control_qubits, self.axes):
            ham *= ctrlham[axis].on(q)

        return ham

    @cached_property
    def tensor(self) -> QubitTensor:
        # FIXME: This approach generates a tensor with unnecessary numerical noise.
        return UnitaryGate.from_hamiltonian(self.hamiltonian, self.qubits).tensor

    def resolve(self, subs: Mapping[str, float]) -> "ControlGate":
        target = self.target.resolve(subs)
        return ControlGate(target, self.control_qubits, self.axes)

    def _diagram_labels_(self) -> List[str]:
        # TODO: Move to config?
        ctrl_labels = {
            "X": "⊖",
            "x": "⊕",
            "Y": "⊘",
            "y": "⊗",
            "z": "○",
            "Z": "●",
        }
        return [ctrl_labels[a] for a in self.axes] + self.target._diagram_labels_()

    def __str__(self) -> str:
        fqubits = " " + " ".join([str(qubit) for qubit in self.control_qubits])
        fparams = str(self.target)

        if self.axes == "Z" * len(self.control_qubits):
            return f"{self.name}({fparams}){fqubits}"
        else:
            return f"{self.name}({fparams}, '{self.axes}'){fqubits}"


# end class ControlGate


class MultiSwapGate(Gate):
    """A permutation of qubits. A generalized multi-qubit Swap."""

    cv_tensor_structure = "swap"
    cv_hermitian = True

    def __init__(self, qubits_in: Qubits, qubits_out: Qubits) -> None:
        if set(qubits_in) != set(qubits_out):
            raise ValueError("Incompatible sets of qubits")

        self.qubits_out = tuple(qubits_out)
        self.qubits_in = tuple(qubits_in)
        super().__init__(qubits=qubits_in)

    @classmethod
    def from_gates(cls, gates: Iterable[Operation]) -> "MultiSwapGate":
        """Create a qubit permutation from a circuit of swap gates"""
        qubits_in = Circuit(gates).qubits
        N = len(qubits_in)

        circ: List[Gate] = []
        for gate in gates:
            if isinstance(gate, Swap):
                circ.append(gate)
            elif isinstance(gate, MultiSwapGate):
                circ.extend(gate.decompose())
            elif isinstance(gate, I) or isinstance(gate, IdentityGate):
                continue
            else:
                raise ValueError("Swap gates must be built from swap gates")

        perm = list(range(N))
        for elem in circ:
            q0, q1 = elem.qubits
            i0 = qubits_in.index(q0)
            i1 = qubits_in.index(q1)
            perm[i1], perm[i0] = perm[i0], perm[i1]

        qubits_out = [qubits_in[p] for p in perm]
        return cls(qubits_in, qubits_out)

    @property
    def H(self) -> "MultiSwapGate":
        return MultiSwapGate(self.qubits_out, self.qubits_in)

    def run(self, ket: State) -> State:
        qubits = ket.qubits
        N = ket.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        tensor = tensors.permute(ket.tensor, perm)

        return State(tensor, qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        qubits = rho.qubits
        N = rho.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)
        perm.extend([idx + N for idx in perm])

        tensor = tensors.permute(rho.tensor, perm)

        return Density(tensor, qubits, rho.memory)

    @cached_property
    def tensor(self) -> QubitTensor:
        N = self.qubit_nb
        qubits = self.qubits

        perm = list(range(2 * N))
        for q0, q1 in zip(qubits, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        U = np.eye(2**N)
        U = np.reshape(U, [2] * 2 * N)
        U = np.transpose(U, perm)
        return tensors.asqutensor(U)


# end class MultiSwapGate


class ReversalGate(MultiSwapGate):
    """A qubit permutation that reverses the order of qubits"""

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits, tuple(reversed(qubits)))


# end class ReversalGate


# DOCME Makes a circular buffer
class CircularShiftGate(MultiSwapGate):
    def __init__(self, qubits: Qubits, shift: int = 1) -> None:
        qubits_in = tuple(qubits)
        nshift = shift % len(qubits)
        qubits_out = qubits_in[nshift:] + qubits_in[:nshift]

        super().__init__(qubits_in, qubits_out)
        self.shift = shift


# end class CircularShiftGate


class QFTGate(Gate):
    """The Quantum Fourier Transform.

    For 3-qubits
    ::
        0: ───H───Z^1/2───Z^1/4───────────────────x───
                  │       │                       │
        1: ───────●───────┼───────H───Z^1/2───────┼───
                          │           │           │
        2: ───────────────●───────────●───────H───x───
    """

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> "InvQFTGate":
        return InvQFTGate(self.qubits)

    @cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class QFTGate


class InvQFTGate(Gate):
    """The inverse Quantum Fourier Transform"""

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> "QFTGate":
        return QFTGate(self.qubits)

    @cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class InvQFTGate


class PauliGate(Gate):
    """
    A Gate corresponding to the exponential of the Pauli algebra element,
    i.e. exp[-1.0j * alpha * element]
    """

    # Kudos: GEC (2019).

    def __init__(self, element: Pauli, alpha: Variable) -> None:
        super().__init__(qubits=element.qubits)
        self.element = element
        self.alpha = alpha

    def __str__(self) -> str:
        return f"PauliGate({self.element}, {self.alpha}) {self.qubits}"

    @property
    def H(self) -> "PauliGate":
        return self**-1

    def __pow__(self, t: Variable) -> "PauliGate":
        return PauliGate(self.element, self.alpha * t)

    @property
    def hamiltonian(self) -> "Pauli":
        return self.alpha * self.element

    def resolve(self, subs: Mapping[str, float]) -> "PauliGate":
        if var.is_symbolic(self.alpha):
            alpha = var.asfloat(self.alpha, subs)
            return PauliGate(self.element, alpha)
        return self

    def decompose(
        self, topology: Optional[nx.Graph] = None
    ) -> Iterator[Union[CNot, XPow, YPow, ZPow]]:
        """
        Returns a Circuit corresponding to the exponential of
        the Pauli algebra element object, i.e. exp[-1.0j * alpha * element]

        If a qubit topology is provided then the returned circuit will
        respect the qubit connectivity, adding swaps as necessary.
        """
        from .translate import translate_PauliGate

        yield from translate_PauliGate(self, topology)

    @cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class PauliGate


# TODO: Move
def merge_diagonal_gates(
    gate0: "DiagonalGate", gate1: "DiagonalGate"
) -> "DiagonalGate":
    qubits = Circuit([gate0, gate1]).qubits
    K = len(qubits)

    extra_qubits0 = tuple(set(qubits) - set(gate0.qubits))
    if extra_qubits0:
        params = np.zeros(shape=[2] * K) + np.resize(
            np.asarray(
                gate0.params,
            ),
            [2] * gate0.qubit_nb,
        )
        gate0 = DiagonalGate(params.flatten(), extra_qubits0 + tuple(gate0.qubits))

    gate0 = gate0.permute(qubits)

    extra_qubits1 = tuple(set(qubits) - set(gate1.qubits))
    if extra_qubits1:
        params = np.zeros(shape=[2] * K) + np.resize(
            np.asarray(
                gate1.params,
            ),
            [2] * gate1.qubit_nb,
        )
        gate1 = DiagonalGate(params.flatten(), extra_qubits1 + tuple(gate1.qubits))
    gate1 = gate1.permute(qubits)

    params = (a + b for a, b in zip(gate0.params, gate1.params))
    return DiagonalGate(params, qubits)


class DiagonalGate(Gate):
    r"""
    A quantum gate whose unitary operator is diagonal in the computational basis.

    .. math::
        \text{DiagonalGate}(u_0, u_1, .., u_{n-1}) =
            \begin{pmatrix}
                \exp(-i h_0)     & 0       & \dots   & 0 \\
                0       & \exp(-i h_1)     & \dots   & 0 \\
                \vdots  & \vdots  & \ddots  & 0 \\
                0       & 0       & \dots   & \exp(-i h_{n-1})
            \end{pmatrix}

    Diagonal gates can be decomposed in O(2^K) 2 qubit gates [1].

    [1] Shende, Bullock, & Markov, Synthesis of Quantum Logic Circuits, 2006
    IEEE Trans. on Computer-Aided Design, 25 (2006), 1000-1010
    `arXiv:0406176 <https://arxiv.org/pdf/quant-ph/0406176.pdf>`_

    """

    cv_tensor_structure = "diagonal"

    def __init__(
        self, diag_hamiltonian: Union[np.ndarray, Sequence[Variable]], qubits: Qubits
    ) -> None:
        super().__init__(qubits=qubits, params=tuple(diag_hamiltonian))

        assert len(self.params) == 2**self.qubit_nb  # FIXME

    # DOCME TESTME
    @classmethod
    def from_gate(cls, gate: Gate) -> "DiagonalGate":
        if isinstance(gate, DiagonalGate):
            return gate

        # Move elsewhere? gate_almost_diagonal?
        def is_diagonal_gate(gate: Gate) -> bool:
            if gate.cv_tensor_structure == "diagonal":
                return True
            if gate.cv_tensor_structure == "identity":
                return True
            return np.allclose(
                np.diag(gate.tensor_diagonal.flatten()), gate.asoperator()
            )

        if not is_diagonal_gate(gate):
            raise ValueError("Not a diagonal gate")

        params = 1.0j * np.log(gate.tensor_diagonal.flatten())
        return cls(list(params), gate.qubits)

    # TESTME with symbolic
    def permute(self, qubits: Qubits) -> "DiagonalGate":
        """Permute the order of the qubits"""
        qubits = tuple(qubits)
        if self.qubits == qubits:
            return self
        params = np.resize(np.asarray(self.params), [2] * self.qubit_nb)
        params = tensors.permute(params, self.qubit_indices(qubits))
        return DiagonalGate(tuple(params.flatten()), qubits)

    @cached_property
    def tensor(self) -> QubitTensor:
        return asqutensor(np.diag(self.tensor_diagonal.flatten()))

    @cached_property
    def tensor_diagonal(self) -> QubitTensor:
        return asqutensor(np.exp(-1.0j * np.asarray(self.params)))

    @property
    def H(self) -> "DiagonalGate":
        return self**-1

    def __pow__(self, e: Variable) -> "DiagonalGate":
        params = [p * e for p in self.params]
        return DiagonalGate(params, self.qubits)


# end class DiagonalGate


# TODO: resolution of variables
class MultiplexedGate(Gate):
    """A uniformly controlled (or multiplexed) gate.

    Given C control qubits and 2^C gate each acting on the same K target qubits,
    return a multiplexed gate with C+K qubits. For each possible bitstring of the
    control bits, a different gate is applied to the target qubits.
    """

    def __init__(self, gates: Sequence[Gate], controls: Qubits) -> None:
        controls = tuple(controls)
        gates = tuple(gates)
        targets = gates[0].qubits
        qubits = tuple(controls) + tuple(targets)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Control and target qubits overlap")

        for gate in gates:
            if gate.qubits != targets:
                raise ValueError("Target qubits of all gates must be the same.")

        if len(gates) != 2 ** len(controls):
            raise ValueError("Wrong number of target gates.")

        super().__init__(qubits)
        self.controls = controls
        self.targets = targets
        self.gates = gates

    @cached_property
    def tensor(self) -> QubitTensor:
        blocks = [gate.asoperator() for gate in self.gates]
        return asqutensor(scipy.linalg.block_diag(*blocks))

    @property
    def H(self) -> "MultiplexedGate":
        return self**-1

    def __pow__(self, e: Variable) -> "MultiplexedGate":
        gates = [gate**e for gate in self.gates]
        return MultiplexedGate(gates, self.controls)

    # TODO: deke to 2^N control gates

    # Testme
    def resolve(self, subs: Mapping[str, float]) -> "MultiplexedGate":
        gates = cast(Sequence[Gate], [gate.resolve(subs) for gate in self.gates])
        return type(self)(gates, self.controls)


# end class MultiplexedGate


class ConditionalGate(MultiplexedGate):
    """A conditional gate.

    Perform gate A on the target qubit if the control qubit is zero,
    else perform gate B. A multiplexed gate with only 1 control.
    """

    def __init__(self, A: Gate, B: Gate, control_qubit: Qubit) -> None:
        super().__init__(gates=[A, B], controls=(control_qubit,))


# end class ConditionalGate


# FIXME: resolve won't work
class MultiplexedRzGate(MultiplexedGate):
    """Uniformly controlled (multiplexed) Rz gate"""

    cv_tensor_structure = "diagonal"

    def __init__(
        self, thetas: Sequence[Variable], controls: Qubits, target: Qubit
    ) -> None:
        thetas = tuple(thetas)
        gates = [Rz(theta, target) for theta in thetas]
        super().__init__(gates=gates, controls=controls)
        self._params = thetas  # FIXME: This seems broken?

    @cached_property
    def tensor(self) -> QubitTensor:
        return asqutensor(np.diag(self.tensor_diagonal.flatten()))

    @cached_property
    def tensor_diagonal(self) -> QubitTensor:
        diagonal: List[float] = []
        for theta in self.params:
            rz = Rz(theta, 0)
            diagonal.extend(np.diag(rz.tensor))

        return asqutensor(diagonal)

    @property
    def H(self) -> "MultiplexedRzGate":
        return self**-1

    def __pow__(self, e: Variable) -> "MultiplexedRzGate":
        thetas = [e * p for p in self.params]
        return MultiplexedRzGate(thetas, self.controls, self.targets[0])


# end class MultiplexedRzGate


class MultiplexedRyGate(MultiplexedGate):
    """Uniformly controlled (multiplexed) Ry gate"""

    def __init__(
        self, thetas: Sequence[Variable], controls: Qubits, target: Qubit
    ) -> None:
        thetas = tuple(thetas)
        gates = [Ry(theta, target) for theta in thetas]
        super().__init__(gates=gates, controls=controls)
        self._params = thetas  # FIXME: This seems broken?

    @property
    def H(self) -> "MultiplexedRyGate":
        return self**-1

    def __pow__(self, e: Variable) -> "MultiplexedRyGate":
        thetas = [e * p for p in self.params]
        return MultiplexedRyGate(thetas, self.controls, self.targets[0])


# end class MultiplexedRyGate


class RandomGate(UnitaryGate):
    r"""Returns a random unitary gate acting on the given qubits.

    Ref:
        "How to generate random matrices from the classical compact groups"
        Francesco Mezzadri, math-ph/0609050
    """

    def __init__(self, qubits: Qubits) -> None:
        qubits = tuple(qubits)
        tensor = utils.unitary_ensemble(2 ** len(qubits))
        super().__init__(tensor, qubits)


# end class RandomGate

# Fin
