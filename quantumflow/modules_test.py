# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import networkx as nx
import numpy as np
import pytest
import scipy.linalg

import quantumflow as qf


def test_identitygate() -> None:
    qubits = [3, 4, 5, 6, 7, 8]
    gate = qf.IdentityGate(qubits)
    ket0 = qf.random_state(qubits)
    ket1 = gate.run(ket0)
    assert qf.states_close(ket0, ket1)

    circ = qf.Circuit(gate.decompose())
    assert len(circ) == 6

    for n in range(1, 6):
        assert qf.IdentityGate(list(range(n))).qubit_nb == n

    assert gate.hamiltonian.is_zero()

    assert gate ** 2 is gate


def test_control_gate() -> None:
    gate0 = qf.ControlGate([0], qf.X(1))
    gate1 = qf.CNot(0, 1)
    assert qf.gates_close(gate0, gate1)

    gateb = qf.ControlGate([1], qf.X(0))
    gate2 = qf.CNot(1, 0)
    assert qf.gates_close(gateb, gate2)

    gate3 = qf.ControlGate([0], qf.Y(1))
    gate4 = qf.CY(0, 1)
    assert qf.gates_close(gate3, gate4)

    gate5 = qf.ControlGate([0], qf.Z(1))
    gate6 = qf.CZ(0, 1)
    assert qf.gates_close(gate5, gate6)

    gate7 = qf.ControlGate([0], qf.H(1))
    gate8 = qf.CH(0, 1)
    assert qf.gates_close(gate7, gate8)

    gate9 = qf.ControlGate([0, 1], qf.X(2))
    gate10 = qf.CCNot(0, 1, 2)
    assert qf.gates_close(gate9, gate10)

    gate11 = qf.ControlGate([0], qf.Swap(1, 2))
    gate12 = qf.CSwap(0, 1, 2)
    assert qf.gates_close(gate11, gate12)


def test_qftgate() -> None:
    circ = qf.Circuit()
    circ += qf.X(2)
    circ += qf.QFTGate([0, 1, 2])

    ket = qf.zero_state(3)
    ket = circ.run(ket)

    true_qft = qf.State(
        [
            0.35355339 + 0.0j,
            0.25000000 + 0.25j,
            0.00000000 + 0.35355339j,
            -0.25000000 + 0.25j,
            -0.35355339 + 0.0j,
            -0.25000000 - 0.25j,
            0.00000000 - 0.35355339j,
            0.25000000 - 0.25j,
        ]
    )

    assert qf.states_close(ket, true_qft)

    assert isinstance(qf.QFTGate([0, 1, 2]).H, qf.InvQFTGate)
    assert isinstance(qf.QFTGate([0, 1, 2]).H.H, qf.QFTGate)

    qf.QFTGate([0, 1, 2]).H.tensor


def test_multiswapgate() -> None:
    # Should be same as a swap.
    perm0 = qf.MultiSwapGate([0, 1], [1, 0])
    gate0 = qf.Swap(0, 1)
    assert qf.gates_close(perm0.asgate(), gate0)
    assert qf.gates_close(perm0.asgate(), perm0.H.asgate())

    perm1 = qf.MultiSwapGate.from_gates(qf.Circuit([gate0]))
    assert qf.gates_close(perm0.asgate(), perm1.asgate())

    perm2 = qf.MultiSwapGate.from_gates(qf.Circuit([perm1]))
    assert qf.gates_close(perm0, perm2)

    with pytest.raises(ValueError):
        qf.MultiSwapGate.from_gates(qf.Circuit(qf.CNot(0, 1)))

    N = 8
    qubits_in = list(range(N))
    qubits_out = np.random.permutation(qubits_in)

    permN = qf.MultiSwapGate(qubits_in, qubits_out)
    assert qf.gates_close(perm0.asgate(), perm1.asgate())
    iden = qf.Circuit([permN, permN.H])
    assert qf.almost_identity(iden.asgate())
    assert qf.circuits_close(iden, qf.Circuit([qf.IdentityGate(qubits_in)]))

    swaps = qf.Circuit(permN.decompose())
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

    rho0 = qf.random_state([0, 1, 3]).asdensity()
    rho1 = perm0.evolve(rho0)
    rho2 = gate0.aschannel().evolve(rho0)
    assert qf.densities_close(rho1, rho2)


def test_reversequbits() -> None:
    rev = qf.ReversalGate([0, 1, 2, 3, 4])
    perm = qf.MultiSwapGate([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    assert qf.circuits_close(qf.Circuit(rev.decompose()), qf.Circuit(perm.decompose()))


def test_rotatequbits() -> None:
    rev = qf.CircularShiftGate([0, 1, 2, 3, 4], 2)
    perm = qf.MultiSwapGate([0, 1, 2, 3, 4], [2, 3, 4, 0, 1])
    assert qf.circuits_close(qf.Circuit(rev.decompose()), qf.Circuit(perm.decompose()))


def test_PauliGate() -> None:
    pauli0 = 0.5 * np.pi * qf.sX(0) * qf.sX(1)

    alpha = 0.4
    circ = qf.PauliGate(pauli0, alpha)
    coords = qf.canonical_coords(circ.asgate())
    assert np.isclose(coords[0], 0.4)

    pauli1 = np.pi * qf.sX(0) * qf.sX(1) * qf.sY(2) * qf.sZ(3)
    _ = qf.PauliGate(pauli1, alpha)

    top2 = nx.star_graph(4)
    pauli2 = 0.5 * np.pi * qf.sX(1) * qf.sY(2) * qf.sZ(3)
    _ = qf.PauliGate(pauli2, alpha).decompose(top2)

    alpha = 0.2
    top3 = nx.star_graph(4)
    pauli3 = 0.5 * np.pi * qf.sX(1) * qf.sX(2)
    circ3 = qf.Circuit(qf.PauliGate(pauli3, alpha).decompose(top3))

    assert qf.circuits_close(circ3, qf.Circuit([qf.I(0), qf.XX(alpha, 1, 2)]))

    qf.PauliGate(qf.sI(0), alpha).decompose(top2)

    with pytest.raises(ValueError):
        pauli4 = 0.5j * np.pi * qf.sX(1) * qf.sX(2)
        _ = qf.Circuit(qf.PauliGate(pauli4, alpha).decompose(top3))

    top4 = nx.DiGraph()
    nx.add_path(top4, [3, 2, 1, 0])
    _ = qf.Circuit(qf.PauliGate(pauli3, alpha).decompose(top4))


def test_PauliGate_more() -> None:

    alphas = [0.1, 2.0, -3.14, -0.4]
    paulis = [
        qf.sZ(0) + 1,
        qf.sY(0),
        qf.sX(0),
        0.5 * np.pi * qf.sZ(0) * qf.sZ(1),
        0.5 * np.pi * qf.sX(0) * qf.sZ(1),
    ]

    for alpha in alphas:
        for pauli in paulis:
            circ = qf.PauliGate(pauli, alpha)
            qbs = circ.qubits
            str(pauli)
            op = pauli.asoperator(qbs)
            U = scipy.linalg.expm(-1.0j * alpha * op)
            gate = qf.Unitary(U, qbs)
            assert qf.gates_close(gate, circ.asgate())

    pauli = qf.sX(0) + qf.sZ(0)
    with pytest.raises(ValueError):
        qf.Circuit(qf.PauliGate(pauli, 0.2).decompose())


def test_PauliGate_resolve() -> None:
    alpha = qf.var.Symbol("alpha")
    g = qf.PauliGate(qf.sZ(0), alpha)
    r = g.resolve(subs={"alpha": 0.3})
    assert r.alpha == 0.3

    g = qf.PauliGate(qf.sZ(0), 0.3)
    assert g.resolve(subs={"alpha": 0.3}) is g


def test_PauliGate_pow() -> None:
    alpha = 0.4
    gate0 = qf.PauliGate(qf.sZ(0), alpha)
    gate1 = gate0 ** 0.3
    gate2 = qf.Unitary(gate0.tensor, gate0.qubits) ** 0.3
    assert qf.gates_close(gate1, gate2)
    assert qf.gates_close(gate1.H, gate2 ** -1)

    gate3 = qf.unitary_from_hamiltonian(gate0.hamiltonian, qubits=gate0.qubits)
    assert qf.gates_close(gate0, gate3)

    s = str(gate0)
    print(s)


# fin
