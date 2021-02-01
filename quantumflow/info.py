# Copyright 2020-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
====================
Information Measures
====================

Measures on vectors, states, and quantum operations.

.. contents:: :local:
.. currentmodule:: quantumflow


Measures in Hilbert Space
##########################

.. autofunction:: fubini_study_angle
.. autofunction:: fubini_study_fidelity
.. autofunction:: fubini_study_close

State Measures
###############

.. autofunction:: state_angle
.. autofunction:: states_close
.. autofunction:: state_fidelity


Mixed state Measures
######################

.. autofunction:: density_angle
.. autofunction:: purity
.. autofunction:: densities_close
.. autofunction:: fidelity
.. autofunction:: bures_distance
.. autofunction:: bures_angle
.. autofunction:: entropy
.. autofunction:: mutual_info


Gate Measures
##############

.. autofunction:: gate_angle
.. autofunction:: gates_close
.. autofunction:: gates_phase_close
.. autofunction:: gates_commute
.. autofunction:: almost_unitary
.. autofunction:: almost_hermitian
.. autofunction:: almost_identity


Channel Measures
#################

.. autofunction:: channel_angle
.. autofunction:: channels_close
.. autofunction:: average_gate_fidelity
.. autofunction:: almost_unital
"""

import numpy as np
import scipy.stats
from scipy.linalg import sqrtm  # matrix square root

from . import tensors
from .channels import Kraus
from .circuits import Circuit
from .config import ATOL
from .modules import IdentityGate
from .ops import Channel, Gate
from .qubits import Qubits
from .states import Density, State, random_state
from .tensors import QubitTensor

__all__ = (
    "fubini_study_angle",
    "fubini_study_fidelity",
    "fubini_study_close",
    "state_fidelity",
    "state_angle",
    "states_close",
    "purity",
    "fidelity",
    "bures_distance",
    "bures_angle",
    "density_angle",
    "densities_close",
    "entropy",
    "mutual_info",
    "gate_angle",
    "channel_angle",
    "gates_close",
    "gates_phase_close",
    "gates_commute",
    "channels_close",
    "circuits_close",
    "trace_distance",
    "average_gate_fidelity",
    "almost_unitary",
    "almost_identity",
    "almost_hermitian",
    "almost_unital",
)


# -- Measures on Hilbert space ---


def fubini_study_angle(vec0: QubitTensor, vec1: QubitTensor) -> QubitTensor:
    """Calculate the Fubini–Study metric between elements of a Hilbert space.

    The Fubini–Study metric is a distance measure between vectors in a
    projective Hilbert space. For gates this space is the Hilbert space of
    operators induced by the Hilbert-Schmidt inner product.
    For 1-qubit rotation gates, Rx, Ry and Rz, this is half the angle (theta)
    in the Bloch sphere.

    The Fubini–Study metric between states is equal to the Burr angle
    between pure states.
    """
    fs_fidelity = fubini_study_fidelity(vec0, vec1)
    return np.arccos(fs_fidelity)


def fubini_study_fidelity(vec0: QubitTensor, vec1: QubitTensor) -> QubitTensor:
    """
    Cosine of the Fubini–Study metric.
    """
    # Suffers from less floating point errors compared to fubini_study_angle

    hs01 = tensors.inner(vec0, vec1)  # Hilbert-Schmidt inner product
    hs00 = tensors.inner(vec0, vec0)
    hs11 = tensors.inner(vec1, vec1)
    ratio = np.absolute(hs01) / np.sqrt(np.absolute(hs00 * hs11))
    fid = np.minimum(ratio, 1.0)  # Compensate for rounding errors.
    return fid


def fubini_study_close(
    vec0: QubitTensor, vec1: QubitTensor, atol: float = ATOL
) -> bool:
    """Return True if vectors are close in the projective Hilbert space.

    Similarity is measured with the Fubini–Study metric.
    """

    return 1 - fubini_study_fidelity(vec0, vec1) <= atol


# -- Measures on pure states ---


def state_fidelity(ket0: State, ket1: State) -> QubitTensor:
    """Return the quantum fidelity between pure states."""
    ket1 = ket1.permute(ket0.qubits)
    tensor = np.absolute(tensors.inner(ket0.tensor, ket1.tensor)) ** 2
    return tensor


def state_angle(ket0: State, ket1: State) -> QubitTensor:
    """The Fubini-Study angle between states.

    Equal to the Burrs angle for pure states.
    """
    ket1 = ket1.permute(ket0.qubits)
    return fubini_study_angle(ket0.tensor, ket1.tensor)


def states_close(ket0: State, ket1: State, atol: float = ATOL) -> bool:
    """Returns True if states are almost identical.

    Closeness is measured with the metric Fubini-Study angle.
    """
    ket1 = ket1.permute(ket0.qubits)

    return fubini_study_close(ket0.tensor, ket1.tensor, atol)


# -- Measures on density matrices ---


def purity(rho: Density) -> float:
    """
    Calculate the purity of a mixed quantum state.

    Purity, defined as tr(rho^2), has an upper bound of 1 for a pure state,
    and a lower bound of 1/D (where D is the Hilbert space dimension) for a
    competently mixed state.

    Two closely related.info are the linear entropy, 1- purity, and the
    participation ratio, 1/purity.
    """
    tensor = rho.tensor
    N = rho.qubit_nb
    matrix = np.reshape(tensor, [2 ** N, 2 ** N])
    return float(np.trace(matrix @ matrix))


def fidelity(rho0: Density, rho1: Density) -> float:
    """Return the fidelity F(rho0, rho1) between two mixed quantum states."""
    rho1 = rho1.permute(rho0.qubits)
    op0 = rho0.asoperator()
    op1 = rho1.asoperator()

    fid = np.real((np.trace(sqrtm(sqrtm(op0) @ op1 @ sqrtm(op0)))) ** 2)
    fid = min(fid, 1.0)
    fid = max(fid, 0.0)  # Correct for rounding errors

    return fid


def bures_distance(rho0: Density, rho1: Density) -> float:
    """Return the Bures distance between mixed quantum states"""
    fid = fidelity(rho0, rho1)
    tr0 = np.trace(rho0.asoperator())
    tr1 = np.trace(rho1.asoperator())

    return np.sqrt(tr0 + tr1 - 2.0 * np.sqrt(fid))


def bures_angle(rho0: Density, rho1: Density) -> float:
    """Return the Bures angle between mixed quantum states"""
    return np.arccos(np.sqrt(fidelity(rho0, rho1)))


def density_angle(rho0: Density, rho1: Density) -> QubitTensor:
    """The Fubini-Study angle between density matrices"""
    rho1 = rho1.permute(rho0.qubits)
    return fubini_study_angle(rho0.tensor, rho1.tensor)


def densities_close(rho0: Density, rho1: Density, atol: float = ATOL) -> bool:
    """Returns True if densities are almost identical.

    Closeness is measured with the Fubini-Study fidelity.
    """
    rho1 = rho1.permute(rho0.qubits)
    return fubini_study_close(rho0.tensor, rho1.tensor, atol)


def entropy(rho: Density, base: float = None) -> float:
    """
    Returns the von-Neumann entropy of a mixed quantum state.

    Args:
        rho:    A density matrix
        base:   Optional logarithm base. Default is base e, and entropy is
               .info in nats. For bits set base to 2.

    Returns:
        The von-Neumann entropy of rho
    """
    op = rho.asoperator()
    probs = np.linalg.eigvalsh(op)
    probs = np.maximum(probs, 0.0)  # Compensate for floating point errors
    return scipy.stats.entropy(probs, base=base)


# TESTME
def mutual_info(
    rho: Density, qubits0: Qubits, qubits1: Qubits = None, base: float = None
) -> float:
    """Compute the bipartite von-Neumann mutual information of a mixed
    quantum state.

    Args:
        rho:    A density matrix of the complete system
        qubits0: Qubits of system 0
        qubits1: Qubits of system 1. If none, taken to be all remaining qubits
        base:   Optional logarithm base. Default is base e

    Returns:
        The bipartite von-Neumann mutual information.
    """
    if qubits1 is None:
        qubits1 = tuple(set(rho.qubits) - set(qubits0))

    rho0 = rho.asdensity(qubits0)
    rho1 = rho.asdensity(qubits1)

    ent = entropy(rho, base)
    ent0 = entropy(rho0, base)
    ent1 = entropy(rho1, base)

    return ent0 + ent1 - ent


# TESTME
def trace_distance(rho0: Density, rho1: Density) -> float:
    """
    Compute the trace distance between two mixed states
    :math:`T(rho_0,rho_1) = (1/2)||rho_0-rho_1||_1`
    """
    op0 = rho0.asoperator()
    op1 = rho1.asoperator()
    return 0.5 * np.linalg.norm(op0 - op1, 1)


# Measures on gates


def gate_angle(gate0: Gate, gate1: Gate) -> QubitTensor:
    """The Fubini-Study angle between gates"""
    gate1 = gate1.permute(gate0.qubits)
    return fubini_study_angle(gate0.tensor, gate1.tensor)


def gates_close(gate0: Gate, gate1: Gate, atol: float = ATOL) -> bool:
    """Returns: True if gates are almost identical, up to
    a phase factor.

    Closeness is measured with the Fubini-Study metric.
    """
    gate1 = gate1.permute(gate0.qubits)
    return fubini_study_close(gate0.tensor, gate1.tensor, atol)


# TESTME
def gates_phase_close(gate0: Gate, gate1: Gate, atol: float = ATOL) -> bool:
    """Returns: True if gates are almost identical and
    have almost the same phase.
    """
    gate1 = gate1.permute(gate0.qubits)
    if not gates_close(gate0, gate1):
        return False
    N = gate0.qubit_nb
    phase = np.trace((gate1 @ gate0.H).asoperator()) / 2 ** N
    return bool(np.isclose(phase, 1.0))


# TESTME
def gates_commute(gate0: Gate, gate1: Gate, atol: float = ATOL) -> bool:
    """Returns: True if gates (almost) commute."""
    gate01 = Circuit([gate0, gate1]).asgate()
    gate10 = Circuit([gate1, gate0]).asgate()
    return gates_close(gate01, gate10, atol)


def almost_unitary(gate: Gate) -> bool:
    """Return true if gate is (almost) unitary"""
    return gates_close(gate @ gate.H, IdentityGate(gate.qubits))


def almost_identity(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) the identity"""
    return gates_close(gate, IdentityGate(gate.qubits))


def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return gates_close(gate, gate.H)


# Measures on circuits

# DOCME
def circuits_close(
    circ0: Circuit, circ1: Circuit, atol: float = ATOL, reps: int = 16
) -> bool:
    """Returns: True if circuits are (probably) almost identical.

    We check closeness by running multiple random initial states
    through both circuits and checking that the resultant states are close.
    """
    qubits = circ0.qubits
    if qubits != circ1.qubits:
        return False

    for _ in range(reps):
        ket = random_state(qubits)
        if not states_close(circ0.run(ket), circ1.run(ket), atol):
            return False
    return True


# Measures on channels


def channel_angle(chan0: Channel, chan1: Channel) -> QubitTensor:
    """The Fubini-Study angle between channels"""
    chan1 = chan1.permute(chan0.qubits)
    return fubini_study_angle(chan0.tensor, chan1.tensor)


def channels_close(chan0: Channel, chan1: Channel, atol: float = ATOL) -> bool:
    """Returns: True if channels are almost identical.

    Closeness is measured with the channel angle.
    """
    chan1 = chan1.permute(chan0.qubits)
    return fubini_study_close(chan0.tensor, chan1.tensor, atol)


# TESTME  multiqubits
def average_gate_fidelity(kraus: Kraus, target: Gate = None) -> QubitTensor:
    """Return the average gate fidelity between a noisy gate (specified by a
    Kraus representation of a superoperator), and a purely unitary target gate.

    If the target gate is not specified, default to identity gate.
    """

    if target is None:
        target = IdentityGate(kraus.qubits)
    else:
        if kraus.qubits != target.qubits:
            raise ValueError("Qubits must be same")  # pragma: no cover  # TESTME

    N = kraus.qubit_nb
    d = 2 ** N

    U = target.H.asoperator()

    summand = 0
    for w, K in zip(kraus.weights, kraus.operators):
        summand += np.absolute(np.trace(w * U @ K.asoperator())) ** 2

    return (d + summand) / (d + d ** 2)


# Author: GEC (2019)
def almost_unital(chan: Channel) -> bool:
    """Return true if the channel is (almost) unital."""
    # Unital channels leave the identity unchanged.
    dim = 2 ** chan.qubit_nb
    eye0 = np.eye(dim, dim)
    rho0 = Density(eye0, chan.qubits)
    rho1 = chan.evolve(rho0)
    eye1 = rho1.asoperator()
    return np.allclose(eye0, eye1)


# fin
