

import quantumflow as qf


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
