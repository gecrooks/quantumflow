# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Visualizations of quantum circuits,
"""

from typing import Any, List
import os
import subprocess
import tempfile

from PIL import Image
import sympy

from .qubits import Qubits
from .gates import P0, P1
from .gates import (I, SWAP, CNOT, CZ, CCNOT, CSWAP, IDEN)

from .ops import Gate
from .stdops import Reset, Measure
from .utils import symbolize, bitlist_to_int, int_to_bitlist
from .circuits import Circuit
from .dagcircuit import DAGCircuit


__all__ = ('LATEX_GATESET',
           'circuit_to_latex',
           'latex_to_image',
           'circuit_to_image',
           'circuit_diagram',
           'circuit_to_diagram')


# TODO: Should be set of types to match GATESET in stdgates?
LATEX_GATESET = frozenset(['I', 'X', 'Y', 'Z', 'H', 'T', 'S', 'T_H', 'S_H',
                           'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'TH', 'CNOT',
                           'CZ', 'SWAP', 'ISWAP', 'PSWAP', 'CCNOT', 'CSWAP',
                           'XX', 'YY', 'ZZ', 'CAN', 'P0', 'P1', 'Reset'])

# TODO: DIAGRAM_GATESET gateset


class NoWire(IDEN):
    """Dummy gate used to draw a gap in a circuit"""
    pass


def circuit_to_latex(
        circ: Circuit,
        qubits: Qubits = None,
        document: bool = True,
        package: str = 'qcircuit',
        options: str = None) -> str:
    """
    Create an image of a quantum circuit in LaTeX.

    Can currently draw X, Y, Z, H, T, S, T_H, S_H, RX, RY, RZ, TX, TY, TZ,
    TH, CNOT, CZ, SWAP, ISWAP, CCNOT, CSWAP, XX, YY, ZZ, CAN, P0 and P1 gates,
    and the Reset operation.

    Args:
        circ:       A quantum Circuit
        qubits:     Optional qubit list to specify qubit order
        document:   If false, just the qcircuit latex is returned. Else the
                    circuit image is wrapped in a standalone LaTeX document
                    ready for typesetting.
        package:    The LaTeX package used for rendering. Either 'qcircuit'
                    (default), or 'quantikz'
    Returns:
        A LaTeX string representation of the circuit.

    Raises:
        NotImplementedError: For unsupported gates.

    Refs:
        LaTeX Qcircuit package
            (https://arxiv.org/pdf/quant-ph/0406003).
        LaTeX quantikz package
            (https://arxiv.org/abs/1809.03842).
    """

    assert package in ['qcircuit', 'quantikz']   # FIXME. Exception

    labels = {
        'S_H': r'S^\dagger',
        'T_H': r'T^\dagger',
        'RX': r'R_x(%s)',
        'RY': r'R_y(%s)',
        'RZ': r'R_z(%s)',
        'TX': r'X^{%s}',
        'TY': r'Y^{%s}',
        'TZ': r'Z^{%s}',
        'TH': r'H^{%s}',
        'XX': r'X\!X^{%s}',
        'YY': r'Y\!Y^{%s}',
        'ZZ': r'Z\!Z^{%s}',
        'ISWAP': r'\text{iSWAP}',
        }

    if qubits is None:
        qubits = circ.qubits
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    layers = _display_layers(circ, qubits)

    layer_code = []

    for layer in layers:
        code = [r'\qw'] * N
        assert isinstance(layer, Circuit)
        for gate in layer:
            idx = [qubit_idx[q] for q in gate.qubits]

            name = gate.name
            params = ''
            if isinstance(gate, Gate) and gate.params:
                params = ','.join(_pretty(p, format='latex')
                                  for p in gate.params.values())

            if name in labels:
                label = labels[name]
            else:
                label = r'\text{%s}' % name
                if params:
                    label += '(%s)'

            if params:
                label = label % params

            # Special 1-qubit gates
            if isinstance(gate, NoWire):
                if package == 'qcircuit':
                    for i in idx:
                        code[i] = r'\push{ }'
                else:  # quantikz
                    for i in idx:
                        code[i] = r''
            elif isinstance(gate, P0):
                code[idx[0]] = r'\push{\ket{0}\!\!\bra{0}} \qw'
            elif isinstance(gate, P1):
                code[idx[0]] = r'\push{\ket{1}\!\!\bra{1}} \qw'
            elif isinstance(gate, Measure):
                code[idx[0]] = r'\meter{}'

            # Special two-qubit gates
            elif isinstance(gate, CNOT):
                code[idx[0]] = r'\ctrl{' + str(idx[1] - idx[0]) + '}'
                code[idx[1]] = r'\targ{}'
            elif isinstance(gate, CZ):
                code[idx[0]] = r'\ctrl{' + str(idx[1] - idx[0]) + '}'
                code[idx[1]] = r'\ctrl{' + str(idx[0] - idx[1]) + '}'
            elif isinstance(gate, SWAP):
                if package == 'qcircuit':
                    code[idx[0]] = r'\qswap \qwx[' + str(idx[1] - idx[0]) + ']'
                    code[idx[1]] = r'\qswap'
                else:  # quantikz
                    code[idx[0]] = r'\swap{' + str(idx[1] - idx[0]) + '}'
                    code[idx[1]] = r'\targX{}'

            # Special three-qubit gates
            elif isinstance(gate, CCNOT):
                code[idx[0]] = r'\ctrl{' + str(idx[1]-idx[0]) + '}'
                code[idx[1]] = r'\ctrl{' + str(idx[2]-idx[1]) + '}'
                code[idx[2]] = r'\targ{}'
            elif isinstance(gate, CSWAP):
                if package == 'qcircuit':
                    code[idx[0]] = r'\ctrl{' + str(idx[1]-idx[0]) + '}'
                    code[idx[1]] = r'\qswap \qwx[' + str(idx[2] - idx[1]) + ']'
                    code[idx[2]] = r'\qswap'
                else:  # quantikz
                    # FIXME: Leaves a weird gap in circuit
                    code[idx[0]] = r'\ctrl{}\vqw{' + str(idx[1]-idx[0]) + '}'
                    code[idx[1]] = r'\swap{' + str(idx[2] - idx[1]) + '}'
                    code[idx[2]] = r'\targX{}'

            # Special multi-qubit gates
            elif isinstance(gate, I):
                pass
            elif isinstance(gate, Reset):
                for i in idx:
                    code[i] = r'\push{\rule{0.1em}{0.5em}\, \ket{0}\,} \qw'

            # Generic 1-qubit gate
            elif(gate.qubit_nb == 1):
                code[idx[0]] = r'\gate{' + label + '}'

            # Generic 2-qubit gate
            elif(gate.qubit_nb == 2 and gate.interchangeable):
                top = min(idx)
                bot = max(idx)

                # TODO: either qubits neigbours, in order,
                # or bit symmetric gate
                if package == 'qcircuit':
                    if bot-top == 1:
                        code[top] = r'\multigate{1}{%s}' % label
                        code[bot] = r'\ghost{%s}' % label
                    else:
                        code[top] = r'\sgate{%s}{%s}' % (label, str(bot - top))
                        code[bot] = r'\gate{%s}' % (label)
                else:  # quantikz
                    if bot-top == 1:
                        code[top] = r'\gate[2]{%s}' % label
                        code[bot] = r''
                    else:
                        code[top] = r'\gate{%s}\vqw{%s}' % (label,
                                                            str(bot - top))
                        code[bot] = r'\gate{%s}' % (label)
            else:
                raise NotImplementedError(str(gate))

        layer_code.append(code)

    code = [r'\qw'] * N
    layer_code.append(code)

    latex_lines = [''] * N

    for line, wire in enumerate(zip(*layer_code)):
        latex = '& ' + ' & '.join(wire)
        if line < N - 1:  # Not last line
            latex += r' \\'
        latex_lines[line] = latex

    if package == 'qcircuit':
        # TODO: qcircuit options
        latex_code = _QCIRCUIT % '\n'.join(latex_lines)
        if document:
            latex_code = _QCIRCUIT_HEADER + latex_code + _QCIRCUIT_FOOTER

    else:  # quantikz
        if options is None:
            options = 'thin lines'
        latex_code = _QUANTIKZ % (options, '\n'.join(latex_lines))
        if document:
            latex_code = _QUANTIKZ_HEADER + latex_code + QUANTIKZ_FOOTER_

    return latex_code


_QCIRCUIT_HEADER = r"""
\documentclass[border={20pt 4pt 20pt 4pt}]{standalone}
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
\begin{document}
"""

_QCIRCUIT = r"""\Qcircuit @C=1.5em @R=1.5em {
%s
}"""

_QCIRCUIT_FOOTER = r"""
\end{document}
"""

_QUANTIKZ_HEADER = r"""
\documentclass[border={20pt 4pt 20pt 4pt}]{standalone}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{quantikz}
\begin{document}
"""

_QUANTIKZ = r"""\begin{quantikz}[%s]
%s
\end{quantikz}
"""

QUANTIKZ_FOOTER_ = r"""
\end{document}
"""


# TODO: Handle unicode via xelatex?

def latex_to_image(latex: str) -> Image:      # pragma: no cover
    """
    Convert a single page LaTeX document into an image.

    To display the returned image, `img.show()`


    Required external dependencies: `pdflatex` (with `qcircuit` package),
    and `poppler` (for `pdftocairo`).

    Args:
        A LaTeX document as a string.

    Returns:
        A PIL Image

    Raises:
        OSError: If an external dependency is not installed.
    """
    tmpfilename = 'circ'
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename)
        with open(tmppath + '.tex', 'w') as latex_file:
            latex_file.write(latex)

        subprocess.run([f"pdflatex",
                        f"-halt-on-error",
                        f"-output-directory={tmpdirname}",
                        f"{tmpfilename}.tex"],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL,
                       check=True)

        subprocess.run(['pdftocairo',
                        '-singlefile',
                        '-png',
                        '-q',
                        tmppath + '.pdf',
                        tmppath])
        img = Image.open(tmppath + '.png')

    return img


def circuit_to_image(circ: Circuit,
                     qubits: Qubits = None) -> Image:   # pragma: no cover
    """Create an image of a quantum circuit.

    A convenience function that calls circuit_to_latex() and latex_to_image().

    Args:
        circ:       A quantum Circuit
        qubits:     Optional qubit list to specify qubit order

    Returns:
        Returns: A PIL Image (Use img.show() to display)

    Raises:
        NotImplementedError: For unsupported gates.
        OSError: If an external dependency is not installed.
    """
    latex = circuit_to_latex(circ, qubits)
    img = latex_to_image(latex)
    return img


def circuit_diagram(
        circ: Circuit,
        qubits: Qubits = None,
        use_unicode: bool = True,
        transpose: bool = False,
        qubit_labels: bool = True) -> str:
    """
    Draw a text diagram of a quantum circuit.

    Can currently draw X, Y, Z, H, T, S, T_H, S_H, RX, RY, RZ, TX, TY, TZ,
    TH, CNOT, CZ, SWAP, ISWAP, CCNOT, CSWAP, XX, YY, ZZ, CAN, P0 and P1 gates,
    and the RESET operation.

    Args:
        circ:           A quantum Circuit
        qubits:         Optional qubit list to specify qubit order
        use_unicode:    If false, use only ASCII characters
        qubit_labels:   If false, do not display qubit names

    Returns:
        A string representation of the circuit.

    Raises:
        NotImplementedError: For unsupported gates.
    """
    # Kudos: Inspired by the circuit diagram drawer from cirq
    # https://github.com/quantumlib/Cirq/blob/master/cirq/circuits/circuit.py#L1435

    if use_unicode:
        TARGET = 'X'
        CTRL = '●'  # ○ ■
        NCTRL = '○'
        CONJ = '⁺'
        BOX_CHARS = STD_BOX_CHARS
        SWAP_TARG = 'x'
    else:
        TARGET = 'X'
        CTRL = '@'
        NCTRL = 'O'
        CONJ = '^-1'
        BOX_CHARS = ASCII_BOX_CHARS
        SWAP_TARG = 'x'

    labels = {
        'NoWire': '  ',
        'P0': '|0><0|',
        'P1': '|1><1|',
        'S_H': 'S' + CONJ,
        'T_H': 'T' + CONJ,
        'V_H': 'V' + CONJ,
        'RX': 'Rx(%s)',
        'RY': 'Ry(%s)',
        'RZ': 'Rz(%s)',
        'TX': 'X^%s',
        'TY': 'Y^%s',
        'TZ': 'Z^%s',
        'TH': 'H^%s',
        'XX': 'XX^%s',
        'YY': 'YY^%s',
        'ZZ': 'ZZ^%s',
        'ISWAP': 'iSWAP',
        'Reset': BOX_CHARS[LEFT+BOT+TOP] + ' <0|',
        }

    multi_labels = {
        'CNOT':  [CTRL, TARGET],
        'CZ':    [CTRL, CTRL],
        'CY':    [CTRL, 'Y'],
        'CV':    [CTRL, 'V'],
        'CV_H':  [CTRL, 'V' + CONJ],
        'CH':    [CTRL, 'H'],
        'SWAP':  [SWAP_TARG, SWAP_TARG],
        'CCNOT': [CTRL, CTRL, TARGET],
        'CSWAP': [CTRL, SWAP_TARG, SWAP_TARG],
        'CCZ':   [CTRL, CTRL, CTRL],
        'CPHASE00': [NCTRL+'(%s)', NCTRL+'(%s)'],
        'CPHASE10': [CTRL+'(%s)', NCTRL+'(%s)'],
        'CPHASE01': [NCTRL+'(%s)', CTRL+'(%s)'],
        'CPHASE': [CTRL+'(%s)', CTRL+'(%s)'],
        'CTX': [CTRL, 'X^%s'],
        'CU3': [CTRL, 'U3(%s)'],
        'CRZ': [CTRL, 'Rz(%s)'],
        }

    def qpad(lines: List[str]) -> List[str]:
        max_length = max(len(l) for l in lines)
        tot_length = max_length+3

        for l, line in enumerate(lines):
            pad_char = [BOX_CHARS[RIGHT+LEFT], ' '][l % 2]
            lines[l] = lines[l].ljust(tot_length, pad_char)
        return lines

    def draw_line(code: List[str], i0: int, i1: int,
                  left_pad: int = 0) -> None:
        i0, i1 = sorted([i0, i1])
        for i in range(i0+1, i1, 2):
            code[i] = (' '*left_pad)+BOX_CHARS[BOT+TOP]
        for i in range(i0+2, i1, 2):
            code[i] = (BOX_CHARS[LEFT+RIGHT]*left_pad)+BOX_CHARS[CROSS]

    if qubits is None:
        qubits = circ.qubits
    N = len(qubits)

    qubit_idx = dict(zip(qubits, range(0, 2*N-1, 2)))
    layers = _display_layers(circ, qubits)
    layer_text = []

    qubit_layer = ['']*(2*N-1)
    if qubit_labels:
        for n in range(N):
            qubit_layer[n*2] = str(qubits[n]) + ': '
        max_length = max(len(l) for l in qubit_layer)
        qubit_layer = [line.ljust(max_length) for line in qubit_layer]
    layer_text.append(qpad(qubit_layer))

    # Gate layers
    for layer in layers:
        code = [''] * (2*N-1)

        assert isinstance(layer, Circuit)
        for gate in layer:
            idx = [qubit_idx[q] for q in gate.qubits]

            name = gate.name
            params = ''
            if isinstance(gate, Gate) and gate.params:
                params = ','.join(_pretty(p, format='text')
                                  for p in gate.params.values())

            if name in labels:
                label = labels[name]
            else:
                label = name
                if params:
                    label += '(%s)'

            if params:
                label = label % params

            # TODO: Are we sure about not showing identity gates?
            if name == 'I':
                pass

            elif name == 'Reset' or name == 'NoWire':
                for i in idx:
                    code[i] = label

            # Special multi-qubit gates
            elif name in multi_labels:
                draw_line(code, min(idx), max(idx))
                for n, mlabel in enumerate(multi_labels[name]):
                    if params and '%' in mlabel:
                        mlabel = mlabel % params
                    code[idx[n]] = mlabel

            # Generic 1-qubit gate
            elif(gate.qubit_nb == 1):
                code[idx[0]] = label

            # Generic 2-qubit gate
            elif(gate.qubit_nb == 2 and gate.interchangeable):
                draw_line(code, min(idx), max(idx), left_pad=len(name)//2)
                code[idx[0]] = label
                code[idx[1]] = label

            else:
                raise NotImplementedError(str(gate))

        layer_text.append(qpad(code))

    circ_text = '\n'.join(''.join(line) for line in zip(*layer_text))

    if transpose:
        boxtrans = dict(zip(BOX_CHARS, _box_char_transpose(BOX_CHARS)))
        circ_text = ''.join(boxtrans.get(c, c) for c in circ_text)
        lines = [list(line) for line in circ_text.splitlines()]
        circ_text = '\n'.join(''.join(c) for c in zip(*lines))

    return circ_text


circuit_to_diagram = circuit_diagram


# ==== UTILITIES ====

def _display_layers(circ: Circuit, qubits: Qubits) -> Circuit:
    """Separate a circuit into groups of gates that do not visually overlap"""
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    gate_layers = DAGCircuit(circ).moments()

    layers = []

    for gl in gate_layers:
        lcirc = Circuit()
        layers.append(lcirc)
        unused = [True] * N
        for gate in gl:
            indices = [qubit_idx[q] for q in gate.qubits]

            if not all(unused[min(indices):max(indices)+1]):
                # New layer
                lcirc = Circuit()
                layers.append(lcirc)
                unused = [True] * N

            unused[min(indices):max(indices)+1] = \
                [False] * (max(indices) - min(indices) + 1)
            lcirc += gate

    return Circuit(layers)


def _pretty(obj: Any, format: str = 'text') -> str:
    """Pretty format an object as a text or latex string."""
    assert format in ['latex', 'text']
    if isinstance(obj, float):
        try:
            if format == 'latex':
                return sympy.latex(symbolize(obj))
            else:
                return str(symbolize(obj)).replace('pi', 'π')
        except ValueError:
            return f'{obj:.4g}'

    out = str(obj)
    if format == 'text':
        out = out.replace('pi', 'π')
    return out


# Unicode and ASCII characters for drawing boxes.
#                  t 0000000011111111
#                  r 0000111100001111
#                  b 0011001100110011
#                  l 0101010101010101
STD_BOX_CHARS     = " ╴╷┐╶─┌┬╵┘│┤└┴├┼"   # noqa: E221
BOLD_BOX_CHARS    = " ╸╻┓╺━┏┳╹┛┃┫┗┻┣╋"   # noqa: E221
DOUBLE_BOX_CHARS  = " ═║╗══╔╦║╝║╣╚╩╠╬"   # noqa: E221  # No half widths
ASCII_BOX_CHARS  = r"   \ -/+ /|+\+++"   # noqa: E221

TOP, RIGHT, BOT, LEFT = 8, 4, 2, 1
CROSS = TOP+RIGHT+BOT+LEFT


def _box_char_transpose(chars: str) -> str:
    return ''.join(chars[bitlist_to_int(list(reversed(int_to_bitlist(n, 4))))]
                   for n in range(16))


# fin
