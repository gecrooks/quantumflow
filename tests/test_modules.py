

import numpy as np
from numpy import pi

import networkx as nx
import scipy.linalg

import pytest

import quantumflow as qf
from quantumflow import backend as bk

from . import ALMOST_ZERO


def test_identitygate():
    qubits = [3, 4, 5, 6, 7, 8]
    gate = qf.IdentityGate(qubits)
    ket0 = qf.random_state(qubits)
    ket1 = gate.run(ket0)
    assert ket0 is ket1

    circ = gate.ascircuit()
    print(circ)
    assert len(circ) == 6


def test_qftgate():
    circ = qf.Circuit()
    circ += qf.X(2)
    circ += qf.QFTGate([0, 1, 2])

    ket = qf.zero_state(3)
    ket = circ.run(ket)

    true_qft = qf.State([0.35355339+0.j, 0.25000000+0.25j,
                         0.00000000+0.35355339j, -0.25000000+0.25j,
                         -0.35355339+0.j, -0.25000000-0.25j,
                         0.00000000-0.35355339j, 0.25000000-0.25j])

    assert qf.states_close(ket, true_qft)


def test_multiswapgate():
    # Should be same as a swap.
    perm0 = qf.MultiSwapGate([0, 1], [1, 0])
    gate0 = qf.SWAP(0, 1)
    assert qf.gates_close(perm0.asgate(), gate0)
    assert qf.gates_close(perm0.asgate(), perm0.H.asgate())

    perm1 = qf.MultiSwapGate.from_gates(qf.Circuit([gate0]))
    assert qf.gates_close(perm0.asgate(), perm1.asgate())

    N = 8
    qubits_in = list(range(N))
    qubits_out = np.random.permutation(qubits_in)

    permN = qf.MultiSwapGate(qubits_in, qubits_out)
    assert qf.gates_close(perm0.asgate(), perm1.asgate())
    iden = qf.Circuit([permN, permN.H]).asgate()
    assert qf.almost_identity(iden)
    assert qf.circuits_close(iden, qf.Circuit([qf.IdentityGate(qubits_in)]))

    swaps = permN.ascircuit()
    # Add identity so we don't lose qubits
    swaps += qf.IdentityGate(permN.qubits_in)
    permN2 = qf.MultiSwapGate.from_gates(swaps)

    assert qf.circuits_close(swaps, qf.Circuit([permN]))
    assert qf.circuits_close(swaps, qf.Circuit([permN2]))
    assert qf.circuits_close(qf.Circuit([permN]), qf.Circuit([permN2]))

    with pytest.raises(ValueError):
        _ = qf.MultiSwapGate([0, 1], [1, 2])

    # Channels
    assert qf.channels_close(perm0.aschannel(), gate0.aschannel())

    rho0 = qf.random_state([0, 1, 3]).asdensity()  # FIXME: qf.random_density()
    rho1 = perm0.evolve(rho0)
    rho2 = gate0.aschannel().evolve(rho0)
    assert qf.densities_close(rho1, rho2)


def test_reversequbits():
    rev = qf.ReversalGate([0, 1, 2, 3, 4])
    perm = qf.MultiSwapGate([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    assert qf.circuits_close(rev.ascircuit(), perm.ascircuit())


def test_rotatequbits():
    rev = qf.CircularShiftGate([0, 1, 2, 3, 4], 2)
    perm = qf.MultiSwapGate([0, 1, 2, 3, 4], [2, 3, 4, 0, 1])
    assert qf.circuits_close(rev.ascircuit(), perm.ascircuit())


def test_PauliGate():
    pauli0 = 0.5 * pi * qf.sX(0) * qf.sX(1)

    alpha = 0.4
    circ = qf.PauliGate(pauli0, alpha)
    print(circ)
    coords = qf.canonical_coords(circ.asgate())
    assert coords[0] - 0.4 == ALMOST_ZERO

    pauli1 = pi * qf.sX(0) * qf.sX(1) * qf.sY(2) * qf.sZ(3)
    circ1 = qf.PauliGate(pauli1, alpha)

    print(pauli1)
    print(circ1)

    print()
    top2 = nx.star_graph(4)
    pauli2 = 0.5 * pi * qf.sX(1) * qf.sY(2) * qf.sZ(3)
    circ2 = qf.PauliGate(pauli2, alpha).ascircuit(top2)
    print(circ2)

    alpha = 0.2
    top3 = nx.star_graph(4)
    pauli3 = 0.5 * pi * qf.sX(1) * qf.sX(2)
    circ3 = qf.PauliGate(pauli3, alpha).ascircuit(top3)

    print(pauli3)
    print(circ3)

    assert qf.circuits_close(circ3, qf.Circuit([qf.I(0), qf.XX(alpha, 1, 2)]))

    qf.PauliGate(qf.sI(0), alpha).ascircuit(top2)

    with pytest.raises(ValueError):
        pauli4 = 0.5j * pi * qf.sX(1) * qf.sX(2)
        _ = qf.PauliGate(pauli4, alpha).ascircuit(top3)

    top4 = nx.DiGraph()
    nx.add_path(top4, [3, 2, 1, 0])
    circ3 = qf.PauliGate(pauli3, alpha).ascircuit(top4)


def test_PauliGate_more():

    alphas = [0.1, 2., -3.14, -0.4]
    paulis = [qf.sZ(0) + 1,
              qf.sY(0),
              qf.sX(0),
              0.5 * pi * qf.sZ(0) * qf.sZ(1),
              0.5 * pi * qf.sX(0) * qf.sZ(1)]

    for alpha in alphas:
        for pauli in paulis:
            print(alpha, pauli)
            circ = qf.PauliGate(pauli, alpha)
            qbs = circ.qubits

            op = bk.evaluate(pauli.asoperator(qbs))
            U = scipy.linalg.expm(-1.0j * alpha * op)
            gate = qf.Unitary(U, *qbs)
            assert qf.gates_close(gate, circ.asgate())

    pauli = qf.sX(0) + qf.sZ(0)
    with pytest.raises(ValueError):
        qf.PauliGate(pauli, 0.2).ascircuit()
