# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.visualization
"""

import os
import shutil
from math import pi

import pytest

import quantumflow as qf

from .stdgates.stdgates_test import _randomize_gate

skip_unless_pdflatex = pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("pdftocairo") is None,
    reason="Necessary external dependencies not installed",
)


def test_circuit_to_latex() -> None:
    qf.circuit_to_latex(qf.ghz_circuit(range(15)))


def test_circuit_to_latex_error() -> None:
    circ = qf.Circuit([qf.RandomGate([0, 1, 2])])
    with pytest.raises(NotImplementedError):
        qf.circuit_to_latex(circ)


def test_visualize_circuit() -> None:
    circ = qf.Circuit()

    circ += qf.I(7)
    circ += qf.X(0)
    circ += qf.Y(1)
    circ += qf.Z(2)
    circ += qf.H(3)
    circ += qf.S(4)
    circ += qf.T(5)
    circ += qf.S_H(6)
    circ += qf.T_H(7)

    circ += qf.Rx(-0.5 * pi, 0)
    circ += qf.Ry(0.5 * pi, 4)
    circ += qf.Rz((1 / 3) * pi, 5)
    circ += qf.Ry(0.222, 6)

    circ += qf.XPow(0.5, 0)
    circ += qf.YPow(0.5, 2)
    circ += qf.ZPow(0.4, 2)
    circ += qf.HPow(0.5, 3)
    circ += qf.ZPow(0.47276, 1)

    # Gate with symbolic parameter
    #  gate = qf.Rz(Symbol('\\theta'), 1)
    # circ += gate

    circ += qf.CNot(1, 2)
    circ += qf.CNot(2, 1)
    # circ += qf.IDEN(*range(8))
    circ += qf.ISwap(4, 2)
    circ += qf.ISwap(6, 5)
    circ += qf.CZ(1, 3)
    circ += qf.Swap(1, 5)

    # circ += qf.Barrier(0, 1, 2, 3, 4, 5, 6)  # Not yet supported in latex

    circ += qf.CCNot(1, 2, 3)
    circ += qf.CSwap(4, 5, 6)

    circ += qf.P0(0)
    circ += qf.P1(1)

    circ += qf.Reset(2)
    circ += qf.Reset(4, 5, 6)

    circ += qf.H(4)

    circ += qf.XX(0.25, 1, 4)
    circ += qf.XX(0.25, 1, 2)
    circ += qf.YY(0.75, 1, 3)
    circ += qf.ZZ(1 / 3, 3, 1)

    circ += qf.CPhase(0, 0, 1)
    circ += qf.CPhase(pi * 1 / 2, 0, 4)

    circ += qf.Can(1 / 3, 1 / 2, 1 / 2, 0, 1)
    circ += qf.Can(1 / 3, 1 / 2, 1 / 2, 2, 4)
    circ += qf.Can(1 / 3, 1 / 2, 1 / 2, 6, 5)

    # circ += qf.Measure(0)
    # circ += qf.Measure(1, 1)

    circ += qf.PSwap(pi / 2, 6, 7)

    circ += qf.Ph(1 / 4, 7)

    circ += qf.CH(1, 6)

    circ += qf.visualization.NoWire([0, 1, 2])
    # circ += qf.visualization.NoWire(4, 1, 2)

    if os.environ.get("QF_VIZTEST"):
        print()
        print(qf.circuit_to_diagram(circ))

    qf.circuit_to_diagram(circ)

    qf.circuit_to_latex(circ)
    qf.circuit_to_latex(circ, package="qcircuit")
    qf.circuit_to_latex(circ, package="quantikz")

    qf.circuit_to_diagram(circ)
    qf.circuit_to_diagram(circ, use_unicode=False)

    latex = qf.circuit_to_latex(circ, package="qcircuit")
    print(latex)
    if os.environ.get("QF_VIZTEST"):
        qf.latex_to_image(latex).show()

    latex = qf.circuit_to_latex(circ, package="quantikz")
    print(latex)

    if os.environ.get("QF_VIZTEST"):
        qf.latex_to_image(latex).show()


def test_circuit_diagram() -> None:
    circ = qf.Circuit()
    circ += qf.Givens(pi, 1, 0)  # Not yet supported in latex
    diag = qf.circuit_to_diagram(circ)
    print()
    print(diag)
    print()


@skip_unless_pdflatex
@pytest.mark.parametrize("gatename", qf.StdGate.cv_stdgates.keys())
def test_stdgates_latex_to_image(gatename: str) -> None:
    if gatename not in qf.LATEX_GATESET:
        print("Latex not supported:", gatename)
        return
    gatet = qf.StdGate.cv_stdgates[gatename]
    gate = _randomize_gate(gatet)
    circ = qf.Circuit(gate)
    latex = qf.circuit_to_latex(circ)
    img = qf.latex_to_image(latex)
    if os.environ.get("QF_VIZTEST"):
        img.show()


@skip_unless_pdflatex
def test_latex_to_image() -> None:
    # TODO: Double check this circuit is correct
    circ = qf.addition_circuit(
        ["a[0]", "a[1]", "a[2]", "a[3]"],
        ["b[0]", "b[1]", "b[2]", "b[3]"],
        ["cin", "cout"],
    )
    order = [
        "cin",
        "a[0]",
        "b[0]",
        "a[1]",
        "b[1]",
        "a[2]",
        "b[2]",
        "a[3]",
        "b[3]",
        "cout",
    ]
    # order = ['cin', 'a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3','cout']

    latex = qf.circuit_to_latex(circ, order)

    qf.latex_to_image(latex)
    if os.environ.get("QF_VIZTEST"):
        qf.latex_to_image(latex).show()


def test_circuit_to_diagram() -> None:
    circ = qf.addition_circuit(
        ["a[0]", "a[1]", "a[2]", "a[3]"],
        ["b[0]", "b[1]", "b[2]", "b[3]"],
        ["cin", "cout"],
    )
    order = [
        "cin",
        "a[0]",
        "b[0]",
        "a[1]",
        "b[1]",
        "a[2]",
        "b[2]",
        "a[3]",
        "b[3]",
        "cout",
    ]
    # order = ['cin', 'a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3','cout']

    text = qf.circuit_to_diagram(circ, order)

    res = """\
cin:  ───────X───●───────────────────────────────────────────────────────────────────●───X───●───
             │   │                                                                   │   │   │   
a[0]: ───●───┼───┼───────────────────●───X───●───X───●───────────────────────────────┼───┼───┼───
         │   │   │                   │   │   │   │   │                               │   │   │   
b[0]: ───X───┼───┼───────────────────┼───●───┼───●───┼───X───────────────────────────┼───┼───┼───
             │   │                   │   │   │   │   │   │                           │   │   │   
a[1]: ───●───┼───┼───────────●───X───X───●───┼───●───X───●───X───●───────────────────┼───┼───┼───
         │   │   │           │   │           │               │   │                   │   │   │   
b[1]: ───X───┼───┼───────────┼───●───────────┼───────────────●───┼───X───────────────┼───┼───┼───
             │   │           │   │           │               │   │   │               │   │   │   
a[2]: ───●───┼───┼───●───X───X───●───────────┼───────────────●───X───●───X───●───────┼───┼───┼───
         │   │   │   │   │                   │                           │   │       │   │   │   
b[2]: ───X───┼───┼───┼───●───────────────────┼───────────────────────────●───┼───X───┼───┼───┼───
             │   │   │   │                   │                           │   │   │   │   │   │   
a[3]: ───●───●───X───X───●───────────────────┼───────────────────────────●───X───●───X───●───┼───
         │       │                           │                                       │       │   
b[3]: ───X───────●───────────────────────────┼───────────────────────────────────────●───────X───
                                             │                                                   
cout: ───────────────────────────────────────X───────────────────────────────────────────────────
"""  # noqa: W291, E501

    print()
    print(text)

    assert res == text

    ascii_circ = qf.circuit_to_diagram(circ, order, use_unicode=False)
    ascii_res = """\
cin:  -------X---@-------------------------------------------------------------------@---X---@---
             |   |                                                                   |   |   |   
a[0]: ---@---+---+-------------------@---X---@---X---@-------------------------------+---+---+---
         |   |   |                   |   |   |   |   |                               |   |   |   
b[0]: ---X---+---+-------------------+---@---+---@---+---X---------------------------+---+---+---
             |   |                   |   |   |   |   |   |                           |   |   |   
a[1]: ---@---+---+-----------@---X---X---@---+---@---X---@---X---@-------------------+---+---+---
         |   |   |           |   |           |               |   |                   |   |   |   
b[1]: ---X---+---+-----------+---@-----------+---------------@---+---X---------------+---+---+---
             |   |           |   |           |               |   |   |               |   |   |   
a[2]: ---@---+---+---@---X---X---@-----------+---------------@---X---@---X---@-------+---+---+---
         |   |   |   |   |                   |                           |   |       |   |   |   
b[2]: ---X---+---+---+---@-------------------+---------------------------@---+---X---+---+---+---
             |   |   |   |                   |                           |   |   |   |   |   |   
a[3]: ---@---@---X---X---@-------------------+---------------------------@---X---@---X---@---+---
         |       |                           |                                       |       |   
b[3]: ---X-------@---------------------------+---------------------------------------@-------X---
                                             |                                                   
cout: ---------------------------------------X---------------------------------------------------
"""  # noqa: W291, E501

    print()
    print(ascii_res)
    assert ascii_res == ascii_circ

    transposed = """\
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ ●─X ●─X ●─X ●─X │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
X─┼─┼─┼─┼─┼─┼─● │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
●─┼─┼─┼─┼─┼─┼─X─● │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ ●─┼─X │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ X─●─● │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ ●─┼─X │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ X─●─● │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ ●─┼─X │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ X─●─● │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ ●─┼─┼─┼─┼─┼─┼─┼─X
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ X─●─● │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ ●─┼─X │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ X─● │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ X─●─● │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ ●─┼─X │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ X─● │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ X─●─● │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ ●─┼─X │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ X─● │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
●─┼─┼─┼─┼─┼─┼─X─● │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
X─┼─┼─┼─┼─┼─┼─● │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
●─┼─┼─┼─┼─┼─┼─┼─X │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │
"""  # noqa

    text = qf.circuit_to_diagram(circ, order, transpose=True, qubit_labels=False)
    print(text)
    assert text == transposed


def test_repr_png_() -> None:
    qf.X(0)._repr_png_()
    qf.Circuit()._repr_png_()
    qf.Circuit([qf.CNot("a", "b")])._repr_png_()


def test_repr_html_() -> None:
    assert qf.X(0)._repr_html_() is not None
    assert qf.Circuit()._repr_html_() is not None
    assert qf.Circuit([qf.CNot("a", "b")])._repr_html_() is not None


def test_gate2_to_diagrams() -> None:
    circ = qf.Circuit()

    circ += qf.CNot(0, 1)
    circ += qf.CZ(0, 1)
    circ += qf.CV(0, 1)
    circ += qf.CV_H(0, 1)
    circ += qf.CH(0, 1)
    circ += qf.Swap(0, 1)
    circ += qf.ISwap(0, 1)

    circ += qf.CNot(0, 2)
    circ += qf.CZ(0, 2)
    circ += qf.CV(0, 2)
    circ += qf.CV_H(0, 2)
    circ += qf.CH(0, 2)
    circ += qf.Swap(0, 2)
    circ += qf.ISwap(0, 2)

    circ += qf.CNot(2, 1)
    circ += qf.CZ(2, 1)
    circ += qf.CV(2, 1)
    circ += qf.CV_H(2, 1)
    circ += qf.CH(2, 1)
    circ += qf.Swap(2, 1)
    circ += qf.ISwap(2, 1)

    print()

    diag = qf.circuit_to_diagram(circ)
    print(diag)

    diag = qf.circuit_to_diagram(circ, use_unicode=False)
    print(diag)

    latex = qf.circuit_to_latex(circ)

    if os.environ.get("QF_VIZTEST"):
        qf.latex_to_image(latex).show()


def test_gate3_to_diagrams() -> None:
    circ = qf.Circuit()
    circ += qf.CCNot(0, 1, 2)
    circ += qf.CCNot(0, 2, 1)
    circ += qf.CSwap(0, 1, 2)
    circ += qf.CSwap(1, 0, 2)
    circ += qf.CCZ(0, 1, 2)
    circ += qf.CCiX(0, 1, 2)
    circ += qf.CCNot(0, 1, 2) ** 0.25
    circ += qf.Deutsch(0.25, 0, 1, 2)

    circ += qf.CV(1, 0)
    circ += qf.CV_H(1, 0)

    print()

    diag = qf.circuit_to_diagram(circ)
    print(diag)
    diag = qf.circuit_to_diagram(circ, use_unicode=False)
    print(diag)

    latex = qf.circuit_to_latex(circ)

    if os.environ.get("QF_VIZTEST"):
        qf.latex_to_image(latex).show()


# fin
