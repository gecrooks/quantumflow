# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Visualizations of quantum circuits,
"""

from typing import Any
import os
import subprocess
import tempfile


from PIL import Image
import sympy

from .qubits import Qubits
from .gates import P0, P1
from .stdgates import (I, SWAP, CNOT, CZ, CCNOT, CSWAP)

from .ops import Gate
from .stdops import Reset, Measure
from .utils import symbolize
from .circuits import Circuit
from .dagcircuit import DAGCircuit


__all__ = ('LATEX_GATESET',
           'circuit_to_latex',
           'render_latex',
           'circuit_to_image')


# TODO: Should be set of types to match GATESET in stdgates?
LATEX_GATESET = frozenset(['I', 'X', 'Y', 'Z', 'H', 'T', 'S', 'T_H', 'S_H',
                           'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ', 'TH', 'CNOT',
                           'CZ', 'SWAP', 'ISWAP', 'PSWAP', 'CCNOT', 'CSWAP',
                           'XX', 'YY', 'ZZ', 'CAN', 'P0', 'P1', 'RESET'])


class NoWire(I):
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
    and the RESET operation.

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

    for layer in layers.elements:
        code = [r'\qw'] * N
        assert isinstance(layer, Circuit)
        for gate in layer:
            idx = [qubit_idx[q] for q in gate.qubits]

            name = gate.name
            params = ''
            if isinstance(gate, Gate) and gate.params:
                params = ','.join(_latex_format(p)
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
            elif(gate.qubit_nb == 2):
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


def _display_layers(circ: Circuit, qubits: Qubits) -> Circuit:
    """Separate a circuit into groups of gates that do not visually overlap"""
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    gate_layers = DAGCircuit(circ).layers()

    layers = []
    lcirc = Circuit()
    layers.append(lcirc)
    unused = [True] * N

    for gl in gate_layers:
        assert isinstance(gl, Circuit)
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


# TODO: Rename to latex_to_image()?
def render_latex(latex: str) -> Image:      # pragma: no cover
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

        subprocess.run(["pdflatex",
                        "-halt-on-error",
                        "-output-directory={}".format(tmpdirname),
                        "{}".format(tmpfilename+'.tex')],
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

    A convenience function that calls circuit_to_latex() and render_latex().

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
    img = render_latex(latex)
    return img


# ==== UTILITIES ====

def _latex_format(obj: Any) -> str:
    """Format an object as a latex string."""
    if isinstance(obj, float):
        try:
            return sympy.latex(symbolize(obj))
        except ValueError:
            return "{0:.4g}".format(obj)

    return str(obj)

# fin
