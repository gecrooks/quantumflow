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
    gate0 = qf.ControlledGate(qf.X(1), [0])
    gate1 = qf.CNot(0, 1)
    assert qf.gates_close(gate0, gate1)

    gateb = qf.ControlledGate(qf.X(0), [1])
    gate2 = qf.CNot(1, 0)
    assert qf.gates_close(gateb, gate2)

    gate3 = qf.ControlledGate(qf.Y(1), [0])
    gate4 = qf.CY(0, 1)
    assert qf.gates_close(gate3, gate4)

    gate5 = qf.ControlledGate(qf.Z(1), [0])
    gate6 = qf.CZ(0, 1)
    assert qf.gates_close(gate5, gate6)

    gate7 = qf.ControlledGate(qf.H(1), [0])
    gate8 = qf.CH(0, 1)
    assert qf.gates_close(gate7, gate8)

    gate9 = qf.ControlledGate(
        qf.X(2),
        [0, 1],
    )
    gate10 = qf.CCNot(0, 1, 2)
    assert qf.gates_close(gate9, gate10)

    gate11 = qf.ControlledGate(qf.Swap(1, 2), [0])
    gate12 = qf.CSwap(0, 1, 2)
    assert qf.gates_close(gate11, gate12)


# def test_controlled_gate_axes() -> None:

#     gate0 = qf.ControlledGate(qf.Z(1), [0], axes="z")
#     gate1 = qf.Circuit([qf.X(0), qf.CZ(0, 1), qf.X(0)]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="ZZ")
#     gate1 = qf.CCNot(0, 1, 2)
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())


#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="zZ")
#     gate1 = qf.Circuit([qf.X(0), qf.CCNot(0, 1, 2), qf.X(0)]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="Zz")
#     gate1 = qf.Circuit([qf.X(1), qf.CCNot(0, 1, 2), qf.X(1)]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="YZ")
#     gate1 = qf.Circuit([qf.V(0), qf.CCNot(0, 1, 2), qf.V(0).H]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="yZ")
#     gate1 = qf.Circuit([qf.V(0).H, qf.CCNot(0, 1, 2), qf.V(0)]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="XZ")
#     gate1 = qf.Circuit([qf.SqrtY(0).H, qf.CCNot(0, 1, 2), qf.SqrtY(0)]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())

#     gate0 = qf.ControlledGate(qf.X(2), [0, 1], axes="xZ")
#     gate1 = qf.Circuit([qf.SqrtY(0), qf.CCNot(0, 1, 2), qf.SqrtY(0).H]).asgate()
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Circuit(gate0.standardize()).asgate())


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
    qubits_out = list(np.random.permutation(qubits_in))

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


def test_DiagonalGate() -> None:

    gate0 = qf.DiagonalGate([0.1, -0.1], qubits=[1])
    gate1 = qf.Rz(0.2, 1)
    assert qf.gates_close(gate0, gate0)
    assert qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0.H, gate0 ** -1)

    gate2 = qf.DiagonalGate([0.1, -0.1], qubits=[1])
    gate3 = qf.ZPow(0.2 / np.pi, 1)
    assert qf.gates_close(gate2, gate3)

    gate4 = qf.DiagonalGate([0, 0, 0, -np.pi], qubits=[0, 1])
    gate5 = qf.CZ(0, 1)
    assert qf.gates_close(gate4, gate5)
    assert qf.gates_close(gate4 ** 0.1, gate5 ** 0.1)

    circ0 = qf.Circuit(gate0.decompose())
    assert qf.gates_close(gate0, circ0.asgate())

    circ4 = qf.Circuit(gate4.decompose())
    assert qf.gates_close(gate4, circ4.asgate())

    gate6 = gate4 ** 0.2
    circ6 = qf.Circuit(gate6.decompose())
    assert qf.gates_close(gate6, circ6.asgate())

    gate7 = qf.DiagonalGate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], qubits=[0, 1, 2])
    circ7 = qf.Circuit(gate7.decompose())
    assert qf.gates_close(gate7, circ7.asgate())

    gate8 = qf.DiagonalGate(tuple(np.random.rand(32)), qubits=[0, 1, 2, 3, 4])
    circ8 = qf.Circuit(gate8.decompose())
    assert qf.gates_close(circ8.asgate(), gate8)

    # circ9 = qf.Circuit([gate8, gate0, gate2])
    assert str(gate2) == "DiagonalGate(1/10, -1/10) 1"


def test_DiagonalGate_from_gate() -> None:
    gate1 = qf.DiagonalGate(tuple(np.random.rand(32)), qubits=[0, 1, 2, 3, 4])
    gate2 = qf.DiagonalGate.from_gate(gate1)
    assert qf.gates_close(gate1, gate2)

    gate3 = qf.ZPow(0.2 / np.pi, 1)
    gate4 = qf.DiagonalGate.from_gate(gate3)
    assert qf.gates_close(gate3, gate4)

    gate5 = qf.Unitary.from_gate(gate1)
    gate6 = qf.DiagonalGate.from_gate(gate5)
    assert qf.gates_close(gate5, gate6)

    gate7 = qf.Swap(0, 1)
    with pytest.raises(ValueError):
        _ = qf.DiagonalGate.from_gate(gate7)

    gate8 = qf.IdentityGate([0, 1, 4])
    gate9 = qf.DiagonalGate.from_gate(gate8)
    assert qf.gates_close(gate8, gate9)


def test_DiagonalGate_permute() -> None:
    gate0 = qf.DiagonalGate([0, 1, 2, 3], qubits=[0, 1])
    gate1 = gate0.permute([1, 0])
    gate2 = gate0.su().permute([1, 0])
    assert qf.gates_close(gate1, gate2)
    assert gate1.params == (0, 2, 1, 3)
    assert gate1.qubits == (1, 0)


def test_merge_diagonal_gates() -> None:
    from quantumflow.modules import merge_diagonal_gates

    gate0 = qf.DiagonalGate([0.01, 0.02, 0.03, 0.04], qubits=[0, 1])
    gate1 = qf.DiagonalGate([0.1, 0.2, 0.3, 0.4], qubits=[1, 2])

    gate2 = merge_diagonal_gates(gate0, gate1)
    gate3 = qf.Circuit([gate0, gate1]).asgate()

    assert qf.gates_close(gate2, gate3)

    gate0 = qf.DiagonalGate([0.01, 0.02, 0.03, 0.04], qubits=[0, 1])
    gate1 = qf.DiagonalGate([0.1, 0.2], qubits=[3])

    gate2 = merge_diagonal_gates(gate0, gate1)
    gate3 = qf.Circuit([gate0, gate1]).asgate()

    assert qf.gates_close(gate2, gate3)


def test_merge_diagonal_gates_symbolic() -> None:
    from sympy import Symbol

    from quantumflow.modules import merge_diagonal_gates

    gate0 = qf.DiagonalGate(
        [Symbol("a0"), Symbol("a1"), Symbol("a2"), Symbol("a3")], qubits=[0, 1]
    )
    gate1 = qf.DiagonalGate(
        [Symbol("b0"), Symbol("b1"), Symbol("b2"), Symbol("b3")], qubits=[1, 2]
    )

    _ = merge_diagonal_gates(gate0, gate1)


def test_DiagonalGate_decomposition_count() -> None:

    for N in range(1, 9):
        qbs = list(range(0, N))
        params = np.random.rand(2 ** N)
        gate = qf.DiagonalGate(params, qbs)
        circ = qf.Circuit(gate.decompose())
        ops = qf.count_operations(circ)
        print(N, ops)
        # print(qf.circuit_to_diagram(circ))

        # From Shende2006a
        if N == 3:
            assert ops[qf.CNot] == 6
            assert ops[qf.Rz] == 7

        # # From "Decomposition of Diagonal Hermitian Quantum Gates Using
        # # Multiple-Controlled Pauli Z Gates"
        # # (2014).
        # # FIXME: Not optimal yet.
        # if N == 4 :
        #     assert ops[qf.CNot] == 14
        #     assert ops[qf.Rz] == 15


def test_MultiplexedRzGate() -> None:
    gate1 = qf.MultiplexedRzGate(thetas=[0.1], controls=(), target=2)
    assert gate1.qubit_nb == 1
    assert qf.gates_close(gate1, qf.Rz(0.1, 2))

    gate2 = qf.MultiplexedRzGate(thetas=[0.1, 0.2], controls=[0], target=1)
    assert gate2.qubit_nb == 2

    gate3 = qf.MultiplexedRzGate(thetas=[0, 0.2], controls=[0], target=1)
    gate4 = qf.CRZ(0.2, 0, 1)
    assert qf.gates_close(gate3, gate4)

    circ1 = qf.Circuit(gate1.decompose())
    assert qf.gates_close(circ1.asgate(), gate1)

    circ2 = qf.Circuit(gate2.decompose())
    assert qf.gates_close(circ2.asgate(), gate2)

    circ3 = qf.Circuit(gate3.decompose())
    assert qf.gates_close(circ3.asgate(), gate3)

    gate4b = qf.MultiplexedRzGate(
        thetas=[0.1, 0.2, 0.3, 0.4], controls=[0, 1], target=2
    )
    circ4b = qf.Circuit(gate4b.decompose())
    assert qf.gates_close(circ4b.asgate(), gate4b)

    gate5 = qf.MultiplexedRzGate(thetas=[0.1, 0.2, 0.3, 0.4], controls=[2, 3], target=0)
    circ5 = qf.Circuit(gate5.decompose())
    assert qf.gates_close(circ5.asgate(), gate5)

    gate6 = qf.MultiplexedRzGate(
        thetas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], controls=[0, 1, 2], target=3
    )
    circ6 = qf.Circuit(gate6.decompose())
    assert qf.gates_close(circ6.asgate(), gate6)

    ops = qf.count_operations(circ6)
    assert ops[qf.CNot] == 12
    assert ops[qf.Rz] == 8
    print(ops)

    gate7 = qf.MultiplexedRzGate(
        thetas=tuple(np.random.rand(16)), controls=[0, 1, 2, 3], target=4
    )
    circ7 = qf.Circuit(gate7.decompose())
    assert qf.gates_close(circ7.asgate(), gate7)

    assert qf.gates_close(gate1.H, gate1 ** -1)

    assert str(gate1) == "MultiplexedRzGate(1/10) 2"


def test_MultiplexedRzGate_str() -> None:
    gate1 = qf.MultiplexedRzGate(thetas=[0.1324], controls=(), target=2)
    s = str(gate1)
    print(s)
    assert s == "MultiplexedRzGate(0.1324) 2"


def test_MultiplexedRyGate() -> None:
    q0, q1 = (0, 1)
    thetas = [0.1, 0.2]
    gate1 = qf.MultiplexedRyGate(thetas, controls=[q0], target=q1)
    assert gate1.qubit_nb == 2

    circ1 = qf.Circuit(
        [
            qf.X(q0),
            qf.ControlledGate(qf.Ry(thetas[0], q1), [q0]),
            qf.X(q0),
            qf.ControlledGate(qf.Ry(thetas[1], q1), [q0]),
        ]
    )
    assert qf.gates_close(gate1, circ1.asgate())

    gate7 = qf.MultiplexedRyGate(
        thetas=tuple(np.random.rand(2)), controls=[0], target=4
    )
    circ7 = qf.Circuit(gate7.decompose())
    assert qf.gates_close(circ7.asgate(), gate7)

    assert qf.gates_close(gate1.H, gate1 ** -1)
    gate2 = qf.Unitary(gate1.asoperator(), gate1.qubits)
    assert qf.gates_close(gate1.H, gate2.H)
    assert qf.gates_close(gate1 ** 2, gate2 ** 2)
    assert qf.gates_close(gate1 ** -1, gate2 ** -1)


def test_MultiplexedGate() -> None:
    gate1 = qf.MultiplexedGate(
        [qf.Rz(0.1, 2), qf.Rz(0.2, 2), qf.Rz(0.3, 2), qf.Rz(0.4, 2)], [0, 1]
    )
    gate2 = qf.MultiplexedRzGate([0.1, 0.2, 0.3, 0.4], [0, 1], 2)
    assert qf.gates_close(gate1, gate2)

    with pytest.raises(ValueError):
        _ = qf.MultiplexedGate(
            [qf.Rz(0.1, 2), qf.Rz(0.2, 2), qf.Rz(0.3, 2), qf.Rz(0.4, 2)], [0, 2]
        )

    with pytest.raises(ValueError):
        _ = qf.MultiplexedGate(
            [qf.Rz(0.1, 2), qf.Rz(0.2, 2), qf.Rz(0.3, 2), qf.Rz(0.4, 2)], [0, 1, 3]
        )

    with pytest.raises(ValueError):
        _ = qf.MultiplexedGate(
            [qf.Rz(0.1, 2), qf.Rz(0.2, 2), qf.Rz(0.3, 2), qf.Rz(0.4, 3)], [0, 1]
        )

    gate3 = gate1.H
    gate4 = qf.UnitaryGate.from_gate(gate1).H
    assert qf.gates_close(gate3, gate4)


def test_ConditionalGate() -> None:
    gate1 = qf.ConditionalGate(qf.H(1), qf.X(1), 0)
    circ2 = qf.Circuit([qf.X(0), qf.CH(0, 1), qf.X(0), qf.CNot(0, 1)])
    assert qf.gates_close(gate1, circ2.asgate())


# fin
