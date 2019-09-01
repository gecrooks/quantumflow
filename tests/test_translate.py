
import quantumflow as qf
from quantumflow.translate import translation_source_gate


def test_translate():
    circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])

    translators = [qf.translate_cswap_to_ccnot,
                   qf.translate_ccnot_to_cnot,
                   qf.translate_cnot_to_cz]
    circ1 = qf.translate(circ0, translators)
    print(circ1)
    # qf.circuit_to_image(circ1).show()
    assert circ1.size() == 33

    circ1 = qf.translate(circ0, translators, recurse=False)

    qf.gates_close(circ0.asgate(), circ1.asgate())


def test_translations():
    from examples.gate_translations import _check_circuit_translations

    _check_circuit_translations()


def test_select_translators():
    targets = [qf.I, qf.TX, qf.TZ, qf.XX]

    trans = qf.select_translators(targets)

    print('>>>>>>', trans)

    sources = [translation_source_gate(t) for t in trans] + targets

    missing = qf.GATESET - set(sources)

    assert len(missing) == 3  # BARENCO, ZYZ, RN
    print()
    print('>>', missing)

    trans2 = qf.select_translators(targets, trans)
    print(trans2)
#

    assert set(trans2) == set(trans)
