
from numpy.random import random
from numpy import pi

import quantumflow as qf


def _assert_translation_close(gate, translator):
    assert qf.gates_close(translator(gate).asgate(), gate)


def test_translate_x():
    _assert_translation_close(qf.X(1), qf.translate_x_to_tx)


def test_translate_y():
    _assert_translation_close(qf.Y(1), qf.translate_y_to_ty)


def test_translate_z():
    _assert_translation_close(qf.Z(1), qf.translate_z_to_tz)


def test_translate_s():
    _assert_translation_close(qf.S(0), qf.translate_s_to_tz)


def test_translate_t():
    _assert_translation_close(qf.T(0), qf.translate_t_to_tz)


def test_translate_invs_to_rz():
    _assert_translation_close(qf.S_H(0), qf.translate_invs_to_tz)


def test_translate_invt_to_rz():
    _assert_translation_close(qf.T(4).H, qf.translate_invt_to_tz)


def test_translate_rx():
    theta = 2*pi*random()
    _assert_translation_close(qf.RX(theta, 4), qf.translate_rx_to_tx)


def test_translate_ry():
    theta = 2*pi*random()
    _assert_translation_close(qf.RY(theta, 4), qf.translate_ry_to_ty)


def test_translate_rz():
    theta = 2*pi*random()
    _assert_translation_close(qf.RZ(theta, 4), qf.translate_rz_to_tz)


def test_translate_tx_to_rx():
    t = 4 * random() - 2
    _assert_translation_close(qf.TX(t, 8), qf.translate_tx_to_rx)


def test_translate_ty_to_ry():
    t = 4 * random() - 2
    _assert_translation_close(qf.TY(t, 0), qf.translate_ty_to_ry)


def test_translate_tz_to_rz():
    t = 4 * random() - 2
    _assert_translation_close(qf.TZ(t, 0), qf.translate_tz_to_rz)


def test_translate_ty_to_xzx():
    t = 4 * random() - 2
    _assert_translation_close(qf.TY(t, 8), qf.translate_ty_to_xzx)


def test_translate_tx_to_zxzxz():
    t = 4 * random() - 2
    _assert_translation_close(qf.TX(t, 0), qf.translate_tx_to_zxzxz)

    # Shortcut
    _assert_translation_close(qf.TX(0.5, 0), qf.translate_tx_to_zxzxz)


def test_translate_hadamard():
    _assert_translation_close(qf.H(0), qf.translate_hadamard_to_zxz)


def test_translate_cnot():
    _assert_translation_close(qf.CNOT(0, 10), qf.translate_cnot_to_cz)


def test_translate_cz():
    _assert_translation_close(qf.CZ(0, 10), qf.translate_cz_to_zz)


def test_translate_iswap():
    _assert_translation_close(qf.ISWAP(0, 10), qf.translate_iswap_to_swap_cz)


def test_translate_swap():
    _assert_translation_close(qf.SWAP(0, 10), qf.translate_swap_to_cnot)


def test_translate_cphase():
    theta = 2 * pi * random()
    _assert_translation_close(qf.CPHASE(theta, 4, 10),
                              qf.translate_cphase_to_zz)


def test_translate_cphase00():
    theta = 2 * pi * random()
    _assert_translation_close(qf.CPHASE00(theta, 4, 10),
                              qf.translate_cphase00_to_zz)


def test_translate_cphase01():
    theta = 2 * pi * random()
    _assert_translation_close(qf.CPHASE01(theta, 4, 10),
                              qf.translate_cphase01_to_zz)


def test_translate_cphase10():
    theta = 2 * pi * random()
    _assert_translation_close(qf.CPHASE10(theta, 4, 10),
                              qf.translate_cphase10_to_zz)


def test_translate_can():
    tx, ty, tz = random(3) * 4 - 2
    _assert_translation_close(qf.CAN(tx, ty, tz, 4, 10),
                              qf.translate_can_to_xx_yy_zz)

    _assert_translation_close(qf.CAN(tx, ty, 0, 4, 10),
                              qf.translate_can_to_xx_yy_zz)

    _assert_translation_close(qf.CAN(0, ty, 0, 4, 10),
                              qf.translate_can_to_xx_yy_zz)

    _assert_translation_close(qf.CAN(tx, 0, 0, 4, 10),
                              qf.translate_can_to_xx_yy_zz)


def test_translate_xx():
    t = 4 * random() - 2
    _assert_translation_close(qf.XX(t, 10, 12), qf.translate_xx_to_zz)


def test_translate_yy():
    t = 4 * random() - 2
    _assert_translation_close(qf.YY(t, 10, 12), qf.translate_yy_to_zz)


def test_translate_zz():
    t = 4 * random() - 2
    _assert_translation_close(qf.ZZ(t, 10, 12), qf.translate_zz_to_cnot)


# def test_translate_piswap():
#     t = 4 * random() - 2
#     _assert_translation_close(qf.PISWAP(t, 10, 12), translate_piswap)


def test_translate_exch():
    t = 4 * random() - 2
    _assert_translation_close(qf.EXCH(t, 10, 12), qf.translate_exch_to_can)


def test_translate_cswap():
    _assert_translation_close(qf.CSWAP(9, 10, 11), qf.translate_cswap_to_ccnot)


def test_translate_ccnot():
    _assert_translation_close(qf.CCNOT(3, 4, 1), qf.translate_ccnot_to_cnot)


def test_translate_circuit():
    circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])

    translators = [qf.translate_cswap_to_ccnot,
                   qf.translate_ccnot_to_cnot,
                   qf.translate_cnot_to_cz]
    circ1 = qf.translate_circuit(circ0, translators)
    print(circ1)
    # qf.circuit_to_image(circ1).show()
    assert circ1.size() == 33

    circ1 = qf.translate_circuit(circ0, translators, recurse=False)


def test_compile():
    circ0 = qf.addition_circuit([0], [1], [2, 3])
    circ1 = qf.compile_circuit(circ0)
    assert qf.circuits_close(circ0, circ1)

    assert circ1.size() == 76

    dagc = qf.DAGCircuit(circ1)
    assert dagc.depth(local=False) == 16
    counts = qf.count_operations(dagc)
    assert counts[qf.TZ] == 27
    assert counts[qf.TX] == 32
    assert counts[qf.CZ] == 17
