
import numpy as np

import pytest

import quantumflow as qf

from . import skip_torch


def test_measure():
    prog = qf.Program()
    prog += qf.Measure(0, ('c', 0))
    prog += qf.Call('X', params=[], qubits=[0])
    prog += qf.Measure(0, ('c', 1))
    ket = prog.run()

    assert ket.qubits == (0,)
    # assert ket.cbits == [('c', 0), ('c', 1)]  # FIXME
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
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 1)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1

    circ = qf.Circuit()
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 0)
    circ += qf.If(qf.X(0), c[1])
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 0
    assert circ.evolve().memory[c[0]] == 0

    circ = qf.Circuit()
    c = qf.Register('c')
    circ += qf.Move(c[0], 0)
    circ += qf.Move(c[1], 0)
    circ += qf.If(qf.X(0), c[1], value=False)
    circ += qf.Measure(0, c[0])
    ket = circ.run()
    assert ket.memory[c[0]] == 1
    assert circ.evolve().memory[c[0]] == 1


def test_project():
    ket0 = qf.zero_state([0, 1, 2])
    ket1 = qf.random_state([2, 3, 4])

    proj = qf.Projection([ket0, ket1])
    assert proj.qubits == (0, 1, 2, 3, 4)

    assert proj.H is proj

    assert proj.H is qf.dagger(proj)


@skip_torch     # FIXME
def test_permutation():
    # Should be same as a swap.
    perm0 = qf.QubitPermutation([0, 1], [1, 0])
    gate0 = qf.SWAP(0, 1)
    assert qf.gates_close(perm0.asgate(), gate0)
    assert qf.gates_close(perm0.asgate(), perm0.H.asgate())

    perm1 = qf.QubitPermutation.from_circuit(qf.Circuit([gate0]))
    assert qf.gates_close(perm0.asgate(), perm1.asgate())

    N = 8
    qubits_in = list(range(N))
    qubits_out = np.random.permutation(qubits_in)
    # print(qubits_out)
    permN = qf.QubitPermutation(qubits_in, qubits_out)
    assert qf.gates_close(perm0.asgate(), perm1.asgate())
    iden = qf.Circuit([permN, permN.H]).asgate()
    assert qf.almost_identity(iden)
    assert qf.circuits_close(iden, qf.Circuit([qf.I(*qubits_in)]))

    swaps = permN.ascircuit()
    swaps += qf.I(*permN.qubits_in)  # Add identity so we don't lose qubits
    permN2 = qf.QubitPermutation.from_circuit(swaps)
    # print(permN2.qubits_in, permN2.qubits_out)

    assert qf.circuits_close(swaps, qf.Circuit([permN]))
    assert qf.circuits_close(swaps, qf.Circuit([permN2]))
    assert qf.circuits_close(qf.Circuit([permN]), qf.Circuit([permN2]))

    with pytest.raises(ValueError):
        _ = qf.QubitPermutation([0, 1], [1, 2])

    # Channels
    assert qf.channels_close(perm0.aschannel(), gate0.aschannel())

    rho0 = qf.random_state([0, 1, 3]).asdensity()  # FIXME: qf.random_density()
    rho1 = perm0.evolve(rho0)
    rho2 = gate0.aschannel().evolve(rho0)
    assert qf.densities_close(rho1, rho2)
