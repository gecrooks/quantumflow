# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.visualization
"""

import os
from math import pi
# from sympy import Symbol

import pytest

import quantumflow as qf

from . import skip_unless_pdflatex


def test_circuit_to_latex():
    qf.circuit_to_latex(qf.ghz_circuit(range(15)))


def test_circuit_to_latex_error():
    circ = qf.Circuit([qf.random_gate([0, 1, 2])])
    with pytest.raises(NotImplementedError):
        qf.circuit_to_latex(circ)


def test_visualize_circuit():
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

    circ += qf.RX(-0.5*pi, 0)
    circ += qf.RY(0.5*pi, 4)
    circ += qf.RZ((1/3)*pi, 5)
    circ += qf.RY(0.222, 6)

    circ += qf.TX(0.5, 0)
    circ += qf.TY(0.5, 2)
    circ += qf.TZ(0.4, 2)
    circ += qf.TH(0.5, 3)
    circ += qf.TZ(0.47276, 1)

    # Gate with symbolic parameter
    #  gate = qf.RZ(Symbol('\\theta'), 1)
    # circ += gate

    circ += qf.CNOT(1, 2)
    circ += qf.CNOT(2, 1)
    # circ += qf.IDEN(*range(8))
    circ += qf.ISWAP(4, 2)
    circ += qf.ISWAP(6, 5)
    circ += qf.CZ(1, 3)
    circ += qf.SWAP(1, 5)

    # circ += qf.Barrier(0, 1, 2, 3, 4, 5, 6)  # Not yet supported in latex

    circ += qf.CCNOT(1, 2, 3)
    circ += qf.CSWAP(4, 5, 6)

    circ += qf.P0(0)
    circ += qf.P1(1)

    circ += qf.Reset(2)
    circ += qf.Reset(4, 5, 6)

    circ += qf.H(4)

    circ += qf.XX(0.25, 1, 4)
    circ += qf.XX(0.25, 1, 2)
    circ += qf.YY(0.75, 1, 3)
    circ += qf.ZZ(1/3, 3, 1)

    circ += qf.CPHASE(0, 0, 1)
    circ += qf.CPHASE(pi*1/2, 0, 4)

    circ += qf.CAN(1/3, 1/2, 1/2, 0, 1)
    circ += qf.CAN(1/3, 1/2, 1/2, 2, 4)
    circ += qf.CAN(1/3, 1/2, 1/2, 6, 5)

    circ += qf.Measure(0)
    circ += qf.Measure(1, 1)

    circ += qf.PSWAP(pi/2, 6, 7)

    circ += qf.Ph(1/4, 7)

    circ += qf.CH(1, 6)

    circ += qf.visualization.NoWire(0, 1, 2)
    # circ += qf.visualization.NoWire(4, 1, 2)

    if os.environ.get('QUANTUMFLOW_VIZTEST'):
        print()
        print(qf.circuit_to_diagram(circ))

    qf.circuit_to_diagram(circ)

    qf.circuit_to_latex(circ)
    qf.circuit_to_latex(circ, package='qcircuit')
    qf.circuit_to_latex(circ, package='quantikz')

    qf.circuit_to_diagram(circ)
    qf.circuit_to_diagram(circ, use_unicode=False)

    latex = qf.circuit_to_latex(circ, package='qcircuit')
    print(latex)
    if os.environ.get('QUANTUMFLOW_VIZTEST'):
        qf.latex_to_image(latex).show()

    latex = qf.circuit_to_latex(circ, package='quantikz')
    print(latex)

    if os.environ.get('QUANTUMFLOW_VIZTEST'):
        qf.latex_to_image(latex).show()


def test_circuit_diagram():
    circ = qf.Circuit()
    circ += qf.Givens(pi, 1, 0)  # Not yet supported in latex
    diag = qf.circuit_to_diagram(circ)
    print()
    print(diag)
    print()


@skip_unless_pdflatex
def test_latex_to_image():
    # TODO: Double check this circuit is correct
    circ = qf.addition_circuit(['a[0]', 'a[1]', 'a[2]', 'a[3]'],
                               ['b[0]', 'b[1]', 'b[2]', 'b[3]'],
                               ['cin', 'cout'])
    order = ['cin', 'a[0]', 'b[0]', 'a[1]', 'b[1]', 'a[2]', 'b[2]', 'a[3]',
             'b[3]', 'cout']
    # order = ['cin', 'a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3','cout']

    latex = qf.circuit_to_latex(circ, order)

    qf.latex_to_image(latex)
    if os.environ.get('QUANTUMFLOW_VIZTEST'):
        qf.latex_to_image(latex).show()


def test_circuit_to_diagram():
    circ = qf.addition_circuit(['a[0]', 'a[1]', 'a[2]', 'a[3]'],
                               ['b[0]', 'b[1]', 'b[2]', 'b[3]'],
                               ['cin', 'cout'])
    order = ['cin', 'a[0]', 'b[0]', 'a[1]', 'b[1]', 'a[2]', 'b[2]', 'a[3]',
             'b[3]', 'cout']
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
"""     # noqa: W291, E501

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
"""     # noqa: W291, E501

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
"""     # noqa

    text = qf.circuit_to_diagram(circ, order,
                                 transpose=True, qubit_labels=False)
    print(text)
    assert text == transposed


def test_repr_png_():
    qf.X(0)._repr_png_()
    qf.Circuit()._repr_png_()
    qf.Circuit([qf.CNOT('a', 'b')])._repr_png_()


def test_repr_html_():
    assert qf.X(0)._repr_html_() is not None
    assert qf.Circuit()._repr_html_() is not None
    assert qf.Circuit([qf.CNOT('a', 'b')])._repr_html_() is not None


# fin
