# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: modules

Multi-qubit gates
#################

Larger unitary computational modules that can be broken up into standard gates.

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

.. autoclass:: ControlledGate
    :members:

.. autoclass:: MultiplexedGate
    :members:

.. autoclass:: ConditionalGate
    :members:

.. autoclass:: MultiplexedRzGate
    :members:

"""

# TODO: Move all to gate module?

from typing import Iterable, Iterator, List, Mapping, Sequence, Union, cast

import networkx as nx
import numpy as np
import scipy
from networkx.algorithms.approximation.steinertree import steiner_tree
from sympy.combinatorics import Permutation

from . import tensors, utils, var
from .circuits import Circuit
from .gates import unitary_from_hamiltonian
from .ops import Gate, Operation, UnitaryGate
from .paulialgebra import Pauli, pauli_commuting_sets, sX, sY, sZ
from .qubits import Qubit, Qubits
from .states import Density, State
from .stdgates import CZ, V_H, CNot, CZPow
from .stdgates import (
    H as H_,
)  # NB: Workaround for name conflict with Gate.H   # FIXME:  needed?
from .stdgates import (
    I,
    Ry,
    Rz,
    # SqrtY,
    # SqrtY_H,
    Swap,
    V,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
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
    "ControlledGate",
    "MultiplexedGate",
    "ConditionalGate",
    "MultiplexedRzGate",
    "MultiplexedRyGate",
    "RandomGate",
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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor(np.eye(2 ** self.qubit_nb))

    def __pow__(self, t: Variable) -> "IdentityGate":
        return self

    @property
    def H(self) -> "IdentityGate":
        return self

    def decompose(self) -> Iterator[I]:  # noqa: E741
        for q in self.qubits:
            yield I(q)


# end class IdentityGate


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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        N = self.qubit_nb
        qubits = self.qubits

        perm = list(range(2 * N))
        for q0, q1 in zip(qubits, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        U = np.eye(2 ** N)
        U = np.reshape(U, [2] * 2 * N)
        U = np.transpose(U, perm)
        return tensors.asqutensor(U)

    def decompose(self) -> Iterator[Swap]:
        """
        Returns a swap network for this permutation, assuming all-to-all
        connectivity.
        """
        qubits = self.qubits

        perm = [self.qubits.index(q) for q in self.qubits_out]
        for idx0, idx1 in Permutation(perm).transpositions():
            yield Swap(qubits[idx0], qubits[idx1])


# end class MultiSwapGate


class ReversalGate(MultiSwapGate):
    """A qubit permutation that reverses the order of qubits"""

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits, tuple(reversed(qubits)))

    def decompose(self) -> Iterator[Swap]:
        qubits = self.qubits
        for idx in range(self.qubit_nb // 2):
            yield Swap(qubits[idx], qubits[-idx - 1])


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
    """The Quantum Fourier Transform circuit.

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

    def decompose(self) -> Iterator[Union[H_, CZPow, Swap]]:
        qubits = self.qubits
        N = len(qubits)
        for n0 in range(N):
            q0 = qubits[n0]
            yield H_(q0)
            for n1 in range(n0 + 1, N):
                q1 = qubits[n1]
                yield CZ(q1, q0) ** (1 / 2 ** (n1 - n0))
        yield from ReversalGate(qubits).decompose()

    @utils.cached_property
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

    def decompose(self) -> Iterator[Union[H_, CZPow, Swap]]:
        gates = list(QFTGate(self.qubits).decompose())
        yield from (gate.H for gate in gates[::-1])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class InvQFTGate


class PauliGate(Gate):
    """
    A Gate corresponding to the exponential of the Pauli algebra element,
    i.e. exp[-1.0j * alpha * element]
    """

    # Kudos: GEC (2019).

    def __init__(self, element: Pauli, alpha: float) -> None:

        super().__init__(qubits=element.qubits)
        self.element = element
        self.alpha = alpha

    def __str__(self) -> str:
        return f"PauliGate({self.element}, {self.alpha}) {self.qubits}"

    @property
    def H(self) -> "PauliGate":
        return self ** -1

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

    # TODO: Move main logic to pauli algebra?
    def decompose(
        self, topology: nx.Graph = None
    ) -> Iterator[Union[CNot, XPow, YPow, ZPow]]:
        """
        Returns a Circuit corresponding to the exponential of
        the Pauli algebra element object, i.e. exp[-1.0j * alpha * element]

        If a qubit topology is provided then the returned circuit will
        respect the qubit connectivity, adding swaps as necessary.
        """
        # Kudos: Adapted from pyquil. The topological network is novel.

        circ = Circuit()
        element = self.element
        alpha = self.alpha

        if element.is_identity() or element.is_zero():
            return circ  # pragma: no cover  # TESTME

        # Check that all terms commute
        groups = pauli_commuting_sets(element)
        if len(groups) != 1:
            raise ValueError("Pauli terms do not all commute")

        for qbs, ops, coeff in element:
            if not np.isclose(complex(coeff).imag, 0.0):
                raise ValueError("Pauli term coefficients must be real")
            theta = complex(coeff).real * alpha

            if len(ops) == 0:
                continue

            # TODO: 1-qubit terms special case

            active_qubits = set()
            change_to_z_basis = Circuit()
            for qubit, op in zip(qbs, ops):
                active_qubits.add(qubit)
                if op == "X":
                    change_to_z_basis += Y(qubit) ** -0.5
                elif op == "Y":
                    change_to_z_basis += X(qubit) ** 0.5

            if topology is not None:
                if not nx.is_directed(topology) or not nx.is_arborescence(topology):
                    # An 'arborescence' is a directed tree
                    active_topology = steiner_tree(topology, active_qubits)
                    center = nx.center(active_topology)[0]
                    active_topology = nx.dfs_tree(active_topology, center)
                else:
                    active_topology = topology
            else:
                active_topology = nx.DiGraph()
                nx.add_path(active_topology, reversed(list(active_qubits)))

            cnot_seq = Circuit()
            order = list(reversed(list(nx.topological_sort(active_topology))))
            for q0 in order[:-1]:
                q1 = list(active_topology.pred[q0])[0]
                if q1 not in active_qubits:
                    cnot_seq += Swap(q0, q1)
                    active_qubits.add(q1)
                else:
                    cnot_seq += CNot(q0, q1)

            circ += change_to_z_basis
            circ += cnot_seq
            circ += Z(order[-1]) ** (2 * theta / np.pi)
            circ += cnot_seq.H
            circ += change_to_z_basis.H
        # end term loop

        yield from circ  # type: ignore

    @utils.cached_property
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

        assert len(self.params) == 2 ** self.qubit_nb  # FIXME

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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return asqutensor(np.diag(self.tensor_diagonal.flatten()))

    @utils.cached_property
    def tensor_diagonal(self) -> QubitTensor:
        return asqutensor(np.exp(-1.0j * np.asarray(self.params)))

    @property
    def H(self) -> "DiagonalGate":
        return self ** -1

    def __pow__(self, e: Variable) -> "DiagonalGate":
        params = [p * e for p in self.params]
        return DiagonalGate(params, self.qubits)

    def decompose(self, topology: nx.Graph = None) -> Iterator[Union[Rz, CNot]]:
        diag_phases = self.params
        N = self.qubit_nb
        qbs = self.qubits

        if N == 1:
            yield Rz((diag_phases[0] - diag_phases[1]), qbs[0])
        else:
            phases = []
            angles = []
            for n in range(0, len(diag_phases), 2):
                phases.append((diag_phases[n] + diag_phases[n + 1]) / 2)
                angles.append(-(diag_phases[n + 1] - diag_phases[n]))

            yield from MultiplexedRzGate(angles, qbs[:-1], qbs[-1]).decompose()
            yield from DiagonalGate(phases, qbs[:-1]).decompose()


# end class DiagonalGate


# TODO: Redo _diagram_labels
# TODO: diagrams
# TODO: Decompose
# ⊖ ⊕ ⊘ ⊗ ● ○
# TODO: resolution of variables
class ControlledGate(Gate):
    """A controlled unitary gate. Given C control qubits and a
    gate acting on K qubits, return a controlled gate with C+K qubits.


    The optional axes argument specifies the basis of the control
    qubits. The length of the sting should be the same as the numebr of control
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

    def __init__(self, gate: Gate, controls: Qubits, axes: str = None) -> None:
        controls = tuple(controls)
        qubits = tuple(controls) + tuple(gate.qubits)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Control and gate qubits overlap")

        if axes is None:
            axes = "Z" * len(controls)
        assert len(axes) == len(controls)

        super().__init__(qubits)
        self.controls = controls
        self.gate = gate
        self.axes = axes

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

        ham = self.gate.hamiltonian
        for q, axis in zip(self.controls, self.axes):
            ham *= ctrlham[axis].on(q)

        return ham

    # # TODO: Rename? Maybe specialize?
    # def standardize(
    #     self,
    # ) -> Iterator[Union["ControlledGate", X, V, V_H, SqrtY, SqrtY_H]]:
    #     """Yield an equivalent controlled gate with standard z-axis controls, pre- and
    #     post-pended with additional 1-qubit gates as necessary"""

    #     transforms = {
    #         "X": SqrtY(0).H,
    #         "x": SqrtY(0),
    #         "Y": V(0),
    #         "y": V(0).H,
    #         "Z": I(0),
    #         "z": X(0),
    #     }

    #     for q, axis in zip(self.controls, self.axes):
    #         if axis != "Z":
    #             yield transforms[axis].on(q)

    #     yield type(self)(self.gate, self.controls)

    #     for q, axis in zip(self.controls, self.axes):
    #         if axis != "Z":
    #             yield transforms[axis].H.on(q)

    # Testme
    def resolve(self, subs: Mapping[str, float]) -> "Gate":
        gate = self.gate.resolve(subs)
        assert isinstance(gate, Gate)
        return type(self)(gate, self.controls, self.axes)

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        # FIXME: This approach generates a tensor with unnecessary numerical noise.
        return unitary_from_hamiltonian(self.hamiltonian, self.qubits).tensor


# end class ControlledGate


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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        blocks = [gate.asoperator() for gate in self.gates]
        return asqutensor(scipy.linalg.block_diag(*blocks))

    @property
    def H(self) -> "MultiplexedGate":
        return self ** -1

    def __pow__(self, e: Variable) -> "MultiplexedGate":
        gates = [gate ** e for gate in self.gates]
        return MultiplexedGate(gates, self.controls)

    # TODO: deke to 2^N control gates

    # Testme
    def resolve(self, subs: Mapping[str, float]) -> "Gate":
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


# FIXME: resolve wont work
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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return asqutensor(np.diag(self.tensor_diagonal.flatten()))

    @utils.cached_property
    def tensor_diagonal(self) -> QubitTensor:
        diagonal = []
        for theta in self.params:
            rz = Rz(theta, 0)
            diagonal.extend(np.diag(rz.tensor))

        return asqutensor(diagonal)

    @property
    def H(self) -> "MultiplexedRzGate":
        return self ** -1

    def __pow__(self, e: Variable) -> "MultiplexedRzGate":
        thetas = [e * p for p in self.params]
        return MultiplexedRzGate(thetas, self.controls, self.targets[0])

    def decompose(self, topology: nx.Graph = None) -> Iterator[Union[Rz, CNot]]:
        thetas = self.params
        N = self.qubit_nb
        controls = self.controls
        target = self.targets[0]

        if N == 1:
            yield Rz(thetas[0], target)
        elif N == 2:
            yield Rz((thetas[0] + thetas[1]) / 2, target)
            yield CNot(controls[0], target)
            yield Rz((thetas[0] - thetas[1]) / 2, target)
            yield CNot(controls[0], target)
        else:
            # FIXME: Not quite optimal.
            # There's additional cancellation of CNOTs that could happen
            # See: From "Decomposition of Diagonal Hermitian Quantum Gates Using
            # Multiple-Controlled Pauli Z Gates" (2014).

            # Note that we lop off 2 qubits with each recursion.
            # This allows us to cancel two cnots by reordering the second
            # deke.

            # If we lopped off one at a time the deke would look like this:
            # t0 = thetas[0: len(thetas) // 2]
            # t1 = thetas[len(thetas) // 2:]
            # thetas0 = [(a + b) / 2 for a, b in zip(t0, t1)]
            # thetas1 = [(a - b) / 2 for a, b in zip(t0, t1)]
            # yield from MultiplexedRzGate(thetas0, qbs[1:]).decompose()
            # yield CNot(qbs[0], qbs[-1])
            # yield from MultiplexedRzGate(thetas1, qbs[1:]).decompose()
            # yield CNot(qbs[0], qbs[-1])

            M = len(thetas) // 4
            quarters = list(thetas[i : i + M] for i in range(0, len(thetas), M))

            theta0 = [(t0 + t1 + t2 + t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
            theta1 = [(t0 - t1 + t2 - t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
            theta2 = [(t0 + t1 - t2 - t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
            theta3 = [(t0 - t1 - t2 + t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]

            yield from MultiplexedRzGate(theta0, controls[2:], target).decompose()
            yield CNot(controls[1], target)
            yield from MultiplexedRzGate(theta1, controls[2:], target).decompose()
            yield CNot(controls[0], target)
            yield from MultiplexedRzGate(theta3, controls[2:], target).decompose()
            yield CNot(controls[1], target)
            yield from MultiplexedRzGate(theta2, controls[2:], target).decompose()
            yield CNot(controls[0], target)


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
        return self ** -1

    def __pow__(self, e: Variable) -> "MultiplexedRyGate":
        thetas = [e * p for p in self.params]
        return MultiplexedRyGate(thetas, self.controls, self.targets[0])

    def decompose(
        self, topology: nx.Graph = None
    ) -> Iterator[Union[V, V_H, MultiplexedRzGate]]:
        thetas = self.params
        controls = self.controls
        target = self.targets[0]

        yield V(target)
        yield MultiplexedRzGate(thetas, controls, target)
        yield V(target).H


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
