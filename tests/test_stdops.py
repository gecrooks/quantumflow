
import numpy as np

import pytest

import quantumflow as qf


def test_measure():
    prog = qf.Circuit()
    prog += qf.Measure(0, ('c', 0))
    prog += qf.X(0)
    prog += qf.Measure(0, ('c', 1))
    ket = prog.run()

    assert ket.qubits == (0,)
    print(ket.memory)
    assert ket.memory[('c', 0)] == 0
    assert ket.memory[('c', 1)] == 1


def test_barrier():
    circ = qf.Circuit()
    circ += qf.Barrier(0, 1, 2)
    circ += qf.Barrier(0, 1, 2).H
    circ.run()
    circ.evolve()

    assert str(qf.Barrier(0, 1, 2)) == 'BARRIER 0 1 2'


def test_if():
    circ = qf.Circuit()
    c = ['c0', 'c1']
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1

    circ = qf.Circuit()
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 0
    assert circ.evolve().memory[c[0]] == 0

    circ = qf.Circuit()
    circ += qf.Store(c[0], 0)
    circ += qf.Store(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1


def test_display_state():
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.StateDisplay(key='state0')

    ket = circ.run()
    assert 'state0' in ket.memory

    rho = circ.evolve()
    assert 'state0' in rho.memory


def test_display_probabilities():
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.ProbabilityDisplay(key='prob')

    ket = circ.run()
    assert 'prob' in ket.memory

    rho = circ.evolve()
    assert 'prob' in rho.memory


def test_density_display():
    circ = qf.Circuit()
    circ += qf.X(1)
    circ += qf.X(2)
    circ += qf.DensityDisplay(key='bloch1', qubits=[1])

    ket = circ.run()
    assert 'bloch1' in ket.memory
    assert ket.memory['bloch1'].qubits == (1,)

    rho = circ.evolve()
    assert 'bloch1' in rho.memory
    assert rho.memory['bloch1'].qubits == (1,)


def test_project():
    ket0 = qf.zero_state([0, 1, 2])
    ket1 = qf.random_state([2, 3, 4])

    proj = qf.Projection([ket0, ket1])
    assert proj.qubits == (0, 1, 2, 3, 4)

    assert proj.H is proj

    assert proj.H is qf.dagger(proj)


def test_permutation():
    # Should be same as a swap.
    perm0 = qf.PermuteQubits([0, 1], [1, 0])
    gate0 = qf.SWAP(0, 1)
    assert qf.gates_close(perm0.asgate(), gate0)
    assert qf.gates_close(perm0.asgate(), perm0.H.asgate())

    perm1 = qf.PermuteQubits.from_circuit(qf.Circuit([gate0]))
    assert qf.gates_close(perm0.asgate(), perm1.asgate())

    N = 8
    qubits_in = list(range(N))
    qubits_out = np.random.permutation(qubits_in)

    permN = qf.PermuteQubits(qubits_in, qubits_out)
    assert qf.gates_close(perm0.asgate(), perm1.asgate())
    iden = qf.Circuit([permN, permN.H]).asgate()
    assert qf.almost_identity(iden)
    assert qf.circuits_close(iden, qf.Circuit([qf.IDEN(*qubits_in)]))

    swaps = permN.ascircuit()
    swaps += qf.IDEN(*permN.qubits_in)  # Add identity so we don't lose qubits
    permN2 = qf.PermuteQubits.from_circuit(swaps)

    assert qf.circuits_close(swaps, qf.Circuit([permN]))
    assert qf.circuits_close(swaps, qf.Circuit([permN2]))
    assert qf.circuits_close(qf.Circuit([permN]), qf.Circuit([permN2]))

    with pytest.raises(ValueError):
        _ = qf.PermuteQubits([0, 1], [1, 2])

    # Channels
    assert qf.channels_close(perm0.aschannel(), gate0.aschannel())

    rho0 = qf.random_state([0, 1, 3]).asdensity()  # FIXME: qf.random_density()
    rho1 = perm0.evolve(rho0)
    rho2 = gate0.aschannel().evolve(rho0)
    assert qf.densities_close(rho1, rho2)


def test_reversequbits():
    rev = qf.ReverseQubits([0, 1, 2, 3, 4])
    perm = qf.PermuteQubits([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    assert qf.circuits_close(rev.ascircuit(), perm.ascircuit())


def test_rotatequbits():
    rev = qf.RotateQubits([0, 1, 2, 3, 4], 2)
    perm = qf.PermuteQubits([0, 1, 2, 3, 4], [2, 3, 4, 0, 1])
    assert qf.circuits_close(rev.ascircuit(), perm.ascircuit())


def test_initialize():
    circ = qf.Circuit()
    circ += qf.H(1)
    ket = qf.random_state([0, 1, 2])
    circ += qf.Initialize(ket)

    assert circ.qubits == (0, 1, 2)
    assert qf.states_close(circ.run(), ket)

    assert qf.states_close(circ.evolve(), ket.asdensity())
