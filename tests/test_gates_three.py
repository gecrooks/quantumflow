
import quantumflow as qf
import numpy as np

from . import ALMOST_ONE


def test_unitary_3qubit():
    assert qf.almost_unitary(qf.CCNOT())
    assert qf.almost_unitary(qf.CSWAP())
    assert qf.almost_unitary(qf.CCZ())


def test_ccnot():
    ket = qf.zero_state(3)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 0, 0] == ALMOST_ONE

    ket = qf.X(1).run(ket)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 1, 0] == ALMOST_ONE

    ket = qf.X(0).run(ket)
    ket = qf.CCNOT(0, 1, 2).run(ket)
    assert ket.vec.asarray()[1, 1, 1] == ALMOST_ONE


def test_cswap():
    ket = qf.zero_state(3)
    ket = qf.X(1).run(ket)
    ket = qf.CSWAP(0, 1, 2).run(ket)
    assert ket.vec.asarray()[0, 1, 0] == ALMOST_ONE

    ket = qf.X(0).run(ket)
    ket = qf.CSWAP(0, 1, 2).run(ket)
    assert ket.vec.asarray()[1, 0, 1] == ALMOST_ONE


def test_ccz():
    ket = qf.zero_state(3)
    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    ket = qf.H(2).run(ket)
    ket = qf.CCZ(0, 1, 2).run(ket)
    ket = qf.H(2).run(ket)
    qf.print_state(ket)
    assert ket.vec.asarray()[1, 1, 1] == ALMOST_ONE

    gate0 = qf.CCZ()
    assert gate0.H is gate0


def test_deutsch():
    gate0 = qf.Deutsch(5*np.pi/2, 0, 1, 2)

    gate1 = qf.CCNOT(0, 1, 2)
    assert qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0, gate1.H)
