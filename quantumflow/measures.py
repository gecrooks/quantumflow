
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
========
Measures
========

QuantumFlow: Measures on vectors, states, and quantum operations.

.. contents:: :local:
.. currentmodule:: quantumflow


Distances in Hilbert Space
##########################

.. autofunction:: inner_product
.. autofunction:: fubini_study_angle
.. autofunction:: vectors_close


State distances
###############

.. autofunction:: state_angle
.. autofunction:: states_close
.. autofunction:: state_fidelity


Mixed state distances
######################

.. autofunction:: density_angle
.. autofunction:: densities_close
.. autofunction:: fidelity
.. autofunction:: bures_distance
.. autofunction:: bures_angle
.. autofunction:: entropy
.. autofunction:: mutual_info


Gate distances
##############

.. autofunction:: gate_angle
.. autofunction:: gates_close
.. autofunction:: gates_phase_close
.. autofunction:: gates_commute


Channel distances
#################

.. autofunction:: channel_angle
.. autofunction:: channels_close
.. autofunction:: diamond_norm
.. autofunction:: average_gate_fidelity

"""
import numpy as np

from scipy.linalg import sqrtm  # matrix square root
import scipy.stats


# from . import backend as bk
from .config import TOLERANCE
from .qubits import Qubits, asarray
from .qubits import vectors_close, fubini_study_angle  # TODO: Move to here
from .states import State, Density, random_state
from .ops import Gate, Channel
from .channels import Kraus
from .circuits import Circuit
from .gates import I, IDEN

from .backends import backend as bk
from .backends import BKTensor


__all__ = ['state_fidelity', 'state_angle', 'states_close',
           'purity', 'fidelity', 'bures_distance', 'bures_angle',
           'density_angle', 'densities_close', 'entropy', 'mutual_info',
           'gate_angle', 'channel_angle', 'gates_close', 'gates_phase_close',
           'gates_commute',  'channels_close',
           'circuits_close', 'diamond_norm',  'trace_distance',
           'average_gate_fidelity',
           'almost_unitary',
           'almost_identity',
           'almost_hermitian',
           ]


# -- Measures on pure states ---

def state_fidelity(state0: State, state1: State) -> BKTensor:
    """Return the quantum fidelity between pure states."""
    assert state0.qubits == state1.qubits   # FIXME

    if bk.BACKEND == 'ctf':     # pragma: no cover
        # Work around for bug in CTF
        # https://github.com/cyclops-community/ctf/issues/62
        tensor = asarray(bk.absolute(bk.inner(state0.tensor,
                                              state1.tensor))) ** 2
    else:
        tensor = bk.absolute(bk.inner(state0.tensor,
                                      state1.tensor)) ** bk.fcast(2)
    return tensor


def state_angle(ket0: State, ket1: State) -> BKTensor:
    """The Fubini-Study angle between states.

    Equal to the Burrs angle for pure states.
    """
    return fubini_study_angle(ket0.vec, ket1.vec)


def states_close(state0: State, state1: State,
                 tolerance: float = TOLERANCE) -> bool:
    """Returns True if states are almost identical.

    Closeness is measured with the metric Fubini-Study angle.
    """
    return vectors_close(state0.vec, state1.vec, tolerance)


# -- Measures on density matrices ---

def purity(rho: Density) -> BKTensor:
    """
    Calculate the purity of a mixed quantum state.

    Purity, defined as tr(rho^2), has an upper bound of 1 for a pure state,
    and a lower bound of 1/D (where D is the Hilbert space dimension) for a
    competently mixed state.

    Two closely related measures are the linear entropy, 1- purity, and the
    participation ratio, 1/purity.
    """
    tensor = rho.tensor
    N = rho.qubit_nb
    matrix = bk.reshape(tensor, [2**N, 2**N])
    return bk.trace(bk.matmul(matrix, matrix))


def fidelity(rho0: Density, rho1: Density) -> float:
    """Return the fidelity F(rho0, rho1) between two mixed quantum states.

    Note: Fidelity cannot be calculated entirely within the tensor backend.
    """
    assert rho0.qubit_nb == rho1.qubit_nb   # FIXME

    rho1 = rho1.permute(rho0.qubits)
    op0 = asarray(rho0.asoperator())
    op1 = asarray(rho1.asoperator())

    fid = np.real((np.trace(sqrtm(sqrtm(op0) @ op1 @ sqrtm(op0)))) ** 2)
    fid = min(fid, 1.0)
    fid = max(fid, 0.0)     # Correct for rounding errors

    return fid


# DOCME
def bures_distance(rho0: Density, rho1: Density) -> float:
    """Return the Bures distance between mixed quantum states

    Note: Bures distance cannot be calculated within the tensor backend.
    """
    fid = fidelity(rho0, rho1)
    op0 = asarray(rho0.asoperator())
    op1 = asarray(rho1.asoperator())
    tr0 = np.trace(op0)
    tr1 = np.trace(op1)

    return np.sqrt(tr0 + tr1 - 2.*np.sqrt(fid))


# DOCME
def bures_angle(rho0: Density, rho1: Density) -> float:
    """Return the Bures angle between mixed quantum states

    Note: Bures angle cannot be calculated within the tensor backend.
    """
    return np.arccos(np.sqrt(fidelity(rho0, rho1)))


def density_angle(rho0: Density, rho1: Density) -> BKTensor:
    """The Fubini-Study angle between density matrices"""
    return fubini_study_angle(rho0.vec, rho1.vec)


def densities_close(rho0: Density, rho1: Density,
                    tolerance: float = TOLERANCE) -> bool:
    """Returns True if densities are almost identical.

    Closeness is measured with the metric Fubini-Study angle.
    """
    return vectors_close(rho0.vec, rho1.vec, tolerance)


def entropy(rho: Density, base: float = None) -> float:
    """
    Returns the von-Neumann entropy of a mixed quantum state.

    Args:
        rho:    A density matrix
        base:   Optional logarithm base. Default is base e, and entropy is
                measures in nats. For bits set base to 2.

    Returns:
        The von-Neumann entropy of rho
    """
    op = asarray(rho.asoperator())
    probs = np.linalg.eigvalsh(op)
    probs = np.maximum(probs, 0.0)  # Compensate for floating point errors
    return scipy.stats.entropy(probs, base=base)


# TESTME
def mutual_info(rho: Density,
                qubits0: Qubits,
                qubits1: Qubits = None,
                base: float = None) -> float:
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


# DOCME TESTME
def trace_distance(rho0: Density, rho1: Density) -> float:
    """
    Compute the trace distance between two mixed states
    :math:`T(rho_0,rho_1) = (1/2)||rho_0-rho_1||_1`

    Note: Trace distance cannot be calculated within the tensor backend.
    """
    op0 = asarray(rho0.asoperator())
    op1 = asarray(rho1.asoperator())
    return 0.5 * np.linalg.norm(op0 - op1, 1)


# Measures on gates

def gate_angle(gate0: Gate, gate1: Gate) -> BKTensor:
    """The Fubini-Study angle between gates"""
    return fubini_study_angle(gate0.vec, gate1.vec)


def gates_close(gate0: Gate, gate1: Gate,
                tolerance: float = TOLERANCE) -> bool:
    """Returns: True if gates are almost identical, up to
    a phase factor.

    Closeness is measured with the gate angle.
    """
    # FIXME: We don't use gate angle! Round off error too large
    return vectors_close(gate0.vec, gate1.vec, tolerance)


# TESTME
def gates_phase_close(gate0: Gate, gate1: Gate,
                      tolerance: float = TOLERANCE) -> bool:
    """Returns: True if gates are almost identical and
    have almost the same phase.
    """
    if not gates_close(gate0, gate1):
        return False
    N = gate0.qubit_nb
    phase = np.trace((gate1 @ gate0.H).asoperator()) / 2**N
    return np.isclose(phase, 1.0)


# TESTME
def gates_commute(gate0: Gate, gate1: Gate,
                  tolerance: float = TOLERANCE) -> bool:
    """Returns: True if gates (almost) commute.
    """
    gate01 = Circuit([gate0, gate1]).asgate()
    gate10 = Circuit([gate1, gate0]).asgate()
    return gates_close(gate01, gate10, tolerance)


# Measures on circuits

# DOCME
def circuits_close(circ0: Circuit, circ1: Circuit,
                   tolerance: float = TOLERANCE,
                   reps: int = 16) -> bool:
    """Returns: True if circuits are (probably) almost identical.

    We check closeness by running multiple random initial states
    through both circuits and checking that the resultant states are close.
    """
    qubits = circ0.qubits
    if qubits != circ1.qubits:
        return False

    for _ in range(reps):
        ket = random_state(qubits)
        if not states_close(circ0.run(ket), circ1.run(ket), tolerance):
            return False
    return True


# Measures on channels

def channel_angle(chan0: Channel, chan1: Channel) -> BKTensor:
    """The Fubini-Study angle between channels"""
    return fubini_study_angle(chan0.vec, chan1.vec)


def channels_close(chan0: Channel, chan1: Channel,
                   tolerance: float = TOLERANCE) -> bool:
    """Returns: True if channels are almost identical.

    Closeness is measured with the channel angle.
    """
    return vectors_close(chan0.vec, chan1.vec, tolerance)


def diamond_norm(chan0: Channel, chan1: Channel) -> float:
    """Return the diamond norm between two completely positive
    trace-preserving (CPTP) superoperators.

    The calculation uses the simplified semidefinite program of Watrous
    [arXiv:0901.4709](http://arxiv.org/abs/0901.4709)
    [J. Watrous, [Theory of Computing 5, 11, pp. 217-238
    (2009)](http://theoryofcomputing.org/articles/v005a011/)]
    """
    # Kudos: Based on MatLab code written by Marcus P. da Silva
    # (https://github.com/BBN-Q/matlab-diamond-norm/)
    import cvxpy as cvx

    if set(chan0.qubits) != set(chan1.qubits):
        raise ValueError('Channels must operate on same qubits')

    if chan0.qubits != chan1.qubits:
        chan1 = chan1.permute(chan0.qubits)

    N = chan0.qubit_nb
    dim = 2**N

    choi0 = asarray(chan0.choi())
    choi1 = asarray(chan1.choi())

    delta_choi = choi0 - choi1

    # Density matrix must be Hermitian, positive semidefinite, trace 1
    rho = cvx.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvx.trace(rho) == 1]

    # W must be Hermitian, positive semidefinite
    W = cvx.Variable([dim**2, dim**2], complex=True)
    constraints += [W == W.H]
    constraints += [W >> 0]

    constraints += [(W - cvx.kron(np.eye(dim), rho)) << 0]

    J = cvx.Parameter([dim**2, dim**2], complex=True)
    objective = cvx.Maximize(cvx.real(cvx.trace(J.H * W)))

    prob = cvx.Problem(objective, constraints)

    J.value = delta_choi
    prob.solve()

    dnorm = prob.value * 2

    # Diamond norm is between 0 and 2. Correct for floating point errors
    dnorm = min(2, dnorm)
    dnorm = max(0, dnorm)

    return dnorm


# TESTME
def average_gate_fidelity(kraus: Kraus, target: Gate = None) -> BKTensor:
    """Return the average gate fidelity between a noisy gate (specified by a
    Kraus representation of a superoperator), and a purely unitary target gate.

    If the target gate is not specified, default to identity gate.
    """

    if target is None:
        target = I(*kraus.qubits)
    else:
        assert kraus.qubits == target.qubits  # FIXME: Exception

    N = kraus.qubit_nb
    d = 2**N

    U = target.H.asoperator()

    summand = 0
    for w, K in zip(kraus.weights, kraus.operators):
        summand += bk.absolute(bk.trace(w * U @ K.asoperator()))**2

    return (d + summand)/(d + d**2)


def almost_unitary(gate: Gate) -> bool:
    """Return true if gate is (almost) unitary"""
    return gates_close(gate @ gate.H, IDEN(*gate.qubits))


def almost_identity(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) the identity"""
    return gates_close(gate, IDEN(*gate.qubits))


def almost_hermitian(gate: Gate) -> bool:
    """Return true if gate tensor is (almost) Hermitian"""
    return gates_close(gate, gate.H)

# fin
