# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Visualizations of quantum circuits,
"""

import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List

import sympy
from PIL import Image
from sympy import Symbol

from . import utils, var
from .circuits import Circuit
from .dagcircuit import DAGCircuit
from .gates import P0, P1
from .modules import IdentityGate
from .ops import Gate, Operation
from .qubits import Qubits
from .stdgates import CZ, CSwap, Swap
from .stdops import Reset

__all__ = (
    "LATEX_GATESET",
    "circuit_to_latex",
    "latex_to_image",
    "circuit_to_image",
    "circuit_to_diagram",
)


# TODO: Should be set of types to match GATESET in stdgates?
LATEX_GATESET = frozenset(
    [
        "I",
        "X",
        "Y",
        "Z",
        "H",
        "T",
        "S",
        "T_H",
        "S_H",
        "V",
        "V_H",
        "Rx",
        "Ry",
        "Rz",
        "SqrtY",
        "SqrtY_H",
        "XPow",
        "YPow",
        "ZPow",
        "HPow",
        "CNot",
        "CZ",
        "Swap",
        "ISwap",
        "PSwap",
        "CV",
        "CV_H",
        "CPhase",
        "CH",
        "Can",
        "CCNot",
        "CSwap",
        "CCZ",
        "CCiX",
        "Deutsch",
        "CCXPow",
        "XX",
        "YY",
        "ZZ",
        "Can",
        "P0",
        "P1",
        "Reset",
        "NoWire",
        "Measure",
        "Ph",
        "ECP",
        "SqrtISwap_H",
        "Barenco",
        "CNotPow",
        "CYPow",
        "CZPow",
        "B",
        "CY",
        "ECP",
    ]
)


kwarg_to_symbol = {
    "alpha": Symbol("α"),
    "lam": Symbol("λ"),
    "nx": Symbol("n_x"),
    "ny": Symbol("n_y"),
    "nz": Symbol("n_z"),
    "p": Symbol("p"),
    "phi": Symbol("φ"),
    "t": Symbol("t"),
    "t0": Symbol("t_0"),
    "t1": Symbol("t_1"),
    "t2": Symbol("t_2"),
    "theta": Symbol("θ"),
    "tx": Symbol("t_x"),
    "ty": Symbol("t_y"),
    "tz": Symbol("t_z"),
    "s": Symbol("s"),
    "b": Symbol("b"),
    "c": Symbol("c"),
}
"""Mapping of standard gate arguments to sympy Symbols"""


class NoWire(IdentityGate):
    """Dummy gate used to draw a gap in a circuit"""

    _diagram_labels = ["  "]


def circuit_to_latex(
    circ: Circuit,
    qubits: Qubits = None,
    document: bool = True,
    package: str = "quantikz",
    options: str = None,
    scale: float = 0.75,
    qubit_labels: bool = True,
) -> str:
    """
    Create an image of a quantum circuit in LaTeX.

    Can currently draw X, Y, Z, H, T, S, T_H, S_H, Rx, Ry, Rz, XPow, YPow, ZPow,
    HPow, CNot, CZ, Swap, ISwap, CCNot, CSwap, XX, YY, ZZ, Can, P0 and P1 gates,
    and the Reset operation.

    Args:
        circ:       A quantum Circuit
        qubits:     Optional qubit list to specify qubit order
        document:   If false, just the qcircuit latex is returned. Else the
                    circuit image is wrapped in a standalone LaTeX document
                    ready for typesetting.
        package:    The LaTeX package used for rendering. Either 'qcircuit'
                    or 'quantikz' (default)
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

    assert package in ["qcircuit", "quantikz"]  # FIXME. Throw Exception

    # TODO: I would be good to move much of the gate dependent details
    # into the gate classes, as has been done for circuit_to_diagram.
    # But there seems to be too many exceptions to be practical. (??)
    from .config import CTRL, TARGET

    latex_labels = {
        "S_H": [r"S^\dagger"],
        "V_H": [r"V^\dagger"],
        "T_H": [r"T^\dagger"],
        "SqrtY": [r"Y^{{\frac{{1}}{{2}}}}"],
        "SqrtY_H": [r"Y^{{-\frac{{1}}{{2}}}}"],
        "Rx": [r"R_x({theta})"],
        "Ry": [r"R_y({theta})"],
        "Rz": [r"R_z({theta})"],
        "XPow": [r"X^{{{t}}}"],
        "YPow": [r"Y^{{{t}}}"],
        "ZPow": [r"Z^{{{t}}}"],
        "HPow": [r"H^{{{t}}}"],
        "XX": [r"X\!X^{{{t}}}"],
        "YY": [r"Y\!Y^{{{t}}}"],
        "ZZ": [r"Z\!Z^{{{t}}}"],
        "ISwap": [r"\text{{iSwap}}"],
        "B": [r"\text{{B}}"],
        "SqrtISwap": [r"\sqrt{{\text{{iSwap}}}}"],
        "SqrtISwap_H": [r"\sqrt{{\text{{iSwap}}}}^\dagger"],
        "CNot": [CTRL, TARGET],
        "CH": [CTRL, "H"],
        "CV": [CTRL, "V"],
        "CV_H": [CTRL, r"V^\dagger"],
        "CNotPow": [CTRL, r"X^{{{t}}}"],
        "CY": [CTRL, r"Y"],
        "CYPow": [CTRL, r"Y^{{{t}}}"],
        "CZPow": [CTRL, r"Z^{{{t}}}"],
        # 3-qubit gates
        "CCNot": [CTRL, CTRL, TARGET],
        "CCXPow": [CTRL, CTRL, r"X^{{{t}}}"],
        "CCZ": [CTRL, CTRL, r"Z"],
        "CCiX": [CTRL, CTRL, r"iX"],
        "Deutsch": [CTRL, CTRL, r"iR^2_x({theta})"],
        "Barenco": [CTRL, r"\text{{Bar}}({phi}, {alpha}, {theta})"],
    }

    if len(circ) == 0:
        # Empty circuit
        circ = Circuit(NoWire([0]))

    if qubits is None:
        qubits = circ.qubits
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    layers = _display_layers(circ, qubits)

    layer_code = []

    if qubit_labels:
        code = [r"\lstick{%s}" % q for q in qubits]
        layer_code.append(code)

    for layer in layers:
        code = [r"\qw"] * N

        for gate in layer:
            elem = gate  # FIXME: elem not gate
            idx = [qubit_idx[q] for q in gate.qubits]

            name = gate.name

            if name not in LATEX_GATESET:
                raise NotImplementedError(str(gate))

            pretty_params: Dict[str, str] = {}
            # FIXME: Do I need gate isinstance (Also below)
            if isinstance(elem, Gate) and elem.params:
                pretty_params = {
                    key: _pretty(value, format="latex")
                    for key, value in zip(gate.cv_args, elem.params)
                }

            # Construct text labels
            name = elem.name

            if name in latex_labels:
                text_labels = latex_labels[name]
                if len(idx) != 1 and len(text_labels) == 1:
                    text_labels = text_labels * len(idx)
                if pretty_params:
                    text_labels = [t.format(**pretty_params) for t in text_labels]
            else:
                if len(name) > 1:
                    name = r"\text{" + name + "}"
                if pretty_params:
                    params_text = ",".join(pretty_params.values())

                    text_labels = [name + "(%s)" % params_text] * len(idx)
                else:
                    text_labels = [name] * len(idx)
                if len(idx) != 1 and not elem.cv_interchangeable:
                    # If not interchangeable, we have to label connections
                    for i in range(elem.qubit_nb):
                        text_labels[i] = text_labels[i] + "_{%s}" % i

            # Special 1-qubit gates
            if isinstance(gate, NoWire):
                if package == "qcircuit":
                    for i in idx:
                        code[i] = r"\push{ }"
                else:  # quantikz
                    for i in idx:
                        code[i] = r""
            elif isinstance(gate, P0):
                code[idx[0]] = r"\push{\ket{0}\!\!\bra{0}} \qw"
            elif isinstance(gate, P1):
                code[idx[0]] = r"\push{\ket{1}\!\!\bra{1}} \qw"
            # elif isinstance(gate, Measure):
            #     code[idx[0]] = r"\meter{}"  # TODO: Add cbit label        # FIXME

            # Special 2-qubit gates
            elif isinstance(gate, CZ):
                code[idx[0]] = r"\ctrl{" + str(idx[1] - idx[0]) + "}"
                code[idx[1]] = r"\ctrl{" + str(idx[0] - idx[1]) + "}"
            elif isinstance(gate, Swap):
                if package == "qcircuit":
                    code[idx[0]] = r"\qswap \qwx[" + str(idx[1] - idx[0]) + "]"
                    code[idx[1]] = r"\qswap"
                else:  # quantikz
                    code[idx[0]] = r"\swap{" + str(idx[1] - idx[0]) + "}"
                    code[idx[1]] = r"\targX{}"

            # interchangeable 2-qubit gate
            elif (
                gate.qubit_nb == 2
                and gate.cv_interchangeable
                and gate.name != "CZPow"
                # and gate.name not in latex_labels
            ):
                top = min(idx)
                bot = max(idx)

                # TODO: either qubits neighbors, in order,
                # or bit symmetric gate
                if package == "qcircuit":
                    if bot - top == 1:
                        code[top] = r"\multigate{1}{%s}" % text_labels[0]
                        code[bot] = r"\ghost{%s}" % text_labels[0]
                    else:
                        code[top] = r"\sgate{%s}{%s}" % (text_labels[0], str(bot - top))
                        code[bot] = r"\gate{%s}" % (text_labels[1])
                else:  # quantikz
                    if bot - top == 1:
                        code[top] = r"\gate[2]{%s}" % text_labels[0]
                        code[bot] = r""
                    else:
                        code[top] = r"\gate{%s}\vqw{%s}" % (
                            text_labels[0],
                            str(bot - top),
                        )
                        code[bot] = r"\gate{%s}" % (text_labels[1])

            # Special three-qubit gates
            elif isinstance(gate, CSwap):
                if package == "qcircuit":
                    code[idx[0]] = r"\ctrl{" + str(idx[1] - idx[0]) + "}"
                    code[idx[1]] = r"\qswap \qwx[" + str(idx[2] - idx[1]) + "]"
                    code[idx[2]] = r"\qswap"
                else:  # quantikz
                    # FIXME: Leaves a weird gap in circuit
                    # code[idx[0]] = r'\ctrl{}\vqw{' + str(idx[1]-idx[0]) + '}'
                    code[idx[0]] = r"\ctrl{" + str(idx[1] - idx[0]) + "}"
                    code[idx[1]] = r"\swap{" + str(idx[2] - idx[1]) + "}"
                    code[idx[2]] = r"\targX{}"

            # Special multi-qubit gates
            elif isinstance(gate, Reset):
                for i in idx:
                    code[i] = r"\push{\rule{0.1em}{0.5em}\, \ket{0}\,} \qw"

            #  Other gates with explicit gate labels
            elif name in latex_labels:
                for i in range(gate.qubit_nb):
                    if text_labels[i] == CTRL:
                        code[idx[i]] = r"\ctrl{" + str(idx[i + 1] - idx[i]) + "}"
                    elif text_labels[i] == TARGET:
                        code[idx[i]] = r"\targ{}"
                    else:
                        code[idx[i]] = r"\gate{" + text_labels[i] + "}"

            # Generic 1-qubit gate
            elif gate.qubit_nb == 1:
                code[idx[0]] = r"\gate{" + text_labels[0] + "}"

            else:
                raise NotImplementedError(str(gate))

        layer_code.append(code)

    code = [r"\qw"] * N
    layer_code.append(code)

    latex_lines = [""] * N

    for line, wire in enumerate(zip(*layer_code)):
        latex = "& " + " & ".join(wire)
        if line < N - 1:  # Not last line
            latex += r" \\"
        latex_lines[line] = latex

    if package == "qcircuit":
        # TODO: qcircuit options
        latex_code = _QCIRCUIT % "\n".join(latex_lines)
        if document:
            latex_code = _QCIRCUIT_HEADER + latex_code + _QCIRCUIT_FOOTER

    else:  # quantikz
        if options is None:
            options = _QUANTIKZ_OPTIONS
        latex_code = _QUANTIKZ % (options, "\n".join(latex_lines))
        latex_code = (r"\adjustbox{scale=%s}{" % scale) + latex_code + "}"

        if document:
            latex_code = _QUANTIKZ_HEADER + latex_code + QUANTIKZ_FOOTER_
    return latex_code


_QCIRCUIT_HEADER = r"""
\documentclass[border={20pt 4pt 20pt 4pt}]{standalone}
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
\usepackage{adjustbox}
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
\usepackage{adjustbox}
\begin{document}
"""

_QUANTIKZ = r"""\begin{quantikz}[%s]
%s
\end{quantikz}
"""

_QUANTIKZ_OPTIONS = "thin lines, column sep=0.75em," "row sep={2.5em,between origins}"

QUANTIKZ_FOOTER_ = r"""
\end{document}
"""


# TODO: Handle unicode via xelatex?


def latex_to_image(latex: str) -> Image:  # pragma: no cover
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
    tmpfilename = "circ"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename)
        with open(tmppath + ".tex", "w") as latex_file:
            latex_file.write(latex)

        subprocess.run(
            [
                "pdflatex",
                "-halt-on-error",
                f"-output-directory={tmpdirname}",
                f"{tmpfilename}.tex",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["pdftocairo", "-singlefile", "-png", "-q", tmppath + ".pdf", tmppath]
        )
        img = Image.open(tmppath + ".png")

    return img


def circuit_to_image(circ: Circuit, qubits: Qubits = None) -> Image:  # pragma: no cover
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


def circuit_to_diagram(
    circ: Circuit,
    qubits: Qubits = None,
    use_unicode: bool = True,
    transpose: bool = False,
    qubit_labels: bool = True,
) -> str:
    """
    Draw a text diagram of a quantum circuit.

    Args:
        circ:           A quantum Circuit
        qubits:         Optional qubit list to specify qubit order
        use_unicode:    If false, return ascii
        qubit_labels:   If false, do not display qubit names

    Returns:
        A string representation of the circuit.
    """
    # Kudos: Inspired by the circuit diagram drawer from cirq
    # https://github.com/quantumlib/Cirq/blob/master/cirq/circuits/circuit.py#L1435

    BOX_CHARS = STD_BOX_CHARS

    def qpad(lines: List[str]) -> List[str]:
        max_length = max(len(k) for k in lines)
        tot_length = max_length + 3

        for k, line in enumerate(lines):
            pad_char = [BOX_CHARS[RIGHT + LEFT], " "][k % 2]
            lines[k] = lines[k].ljust(tot_length, pad_char)
        return lines

    def draw_line(code: List[str], i0: int, i1: int, left_pad: int = 0) -> None:
        i0, i1 = sorted([i0, i1])
        for i in range(i0 + 1, i1, 2):
            code[i] = (" " * left_pad) + BOX_CHARS[BOT + TOP]
        for i in range(i0 + 2, i1, 2):
            code[i] = (BOX_CHARS[LEFT + RIGHT] * left_pad) + BOX_CHARS[CROSS]

    if len(circ) == 0:
        # Empty circuit
        circ = Circuit(NoWire([0]))

    if qubits is None:
        qubits = circ.qubits
    N = len(qubits)

    qubit_idx = dict(zip(qubits, range(0, 2 * N - 1, 2)))
    layers = _display_layers(circ, qubits)
    layer_text = []

    qubit_layer = [""] * (2 * N - 1)
    if qubit_labels:
        for n in range(N):
            qubit_layer[n * 2] = str(qubits[n]) + ": "
        max_length = max(len(k) for k in qubit_layer)
        qubit_layer = [line.ljust(max_length) for line in qubit_layer]
    layer_text.append(qpad(qubit_layer))

    for layer in layers:
        code = [""] * (2 * N - 1)

        for elem in layer:
            idx = [qubit_idx[q] for q in elem.qubits]

            # Pretty print parameters
            pretty_params: Dict[str, str] = {}
            params = elem.params
            if params:
                pretty_params = {
                    key: _pretty(value, format="text")
                    for key, value in zip(elem.cv_args, params)
                }

            # Construct text labels
            name = elem.name
            if elem._diagram_labels:
                text_labels = elem._diagram_labels
                if len(idx) != 1 and len(text_labels) == 1:
                    text_labels = list(text_labels) * len(idx)
                text_labels = [t.format(**pretty_params) for t in text_labels]
            else:
                if pretty_params:
                    params_text = ",".join(pretty_params.values())
                    text_labels = [name + "(%s)" % params_text] * len(idx)
                else:
                    text_labels = [name] * len(idx)
                if len(idx) != 1 and not elem.cv_interchangeable:
                    # If not interchangeable, we have to label connections
                    for i in range(elem.qubit_nb):
                        text_labels[i] = text_labels[i] + "_%s" % i

            if not use_unicode:
                text_labels = [_unicode_to_ascii(tl) for tl in text_labels]

            if elem.qubit_nb != 1 and not elem._diagram_noline:
                pad = len(re.split(r"[_^(]+", text_labels[0])[0]) // 2
                draw_line(code, min(idx), max(idx), left_pad=pad)

            for i in range(elem.qubit_nb):
                code[idx[i]] = text_labels[i]
        # end loop over elements

        layer_text.append(qpad(code))
    # end loop over layers

    circ_text = "\n".join("".join(line) for line in zip(*layer_text))

    if transpose:
        boxtrans = dict(zip(BOX_CHARS, _box_char_transpose(BOX_CHARS)))
        circ_text = "".join(boxtrans.get(c, c) for c in circ_text)
        lines = [list(line) for line in circ_text.splitlines()]
        circ_text = "\n".join("".join(c) for c in zip(*lines))

    if not use_unicode:
        circ_text = _unicode_to_ascii(circ_text)

    return circ_text + "\n"


# ==== UTILITIES ====


def _display_layers(circ: Circuit, qubits: Qubits) -> Circuit:
    """Separate a circuit into groups of gates that do not visually overlap"""

    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))

    # Split circuit into Moments, where the elements in each
    # moment operate on non-overlapping qubits
    gate_layers = DAGCircuit(circ).moments()
    # print(len(gate_layers))

    # Now split each moment into visual layers where the
    # control lines do not visually overlap.
    layers = []
    for gl in gate_layers:
        lcirc: List[Operation] = []
        layers.append(lcirc)
        unused = [True] * N
        for gate in gl:
            indices = [qubit_idx[q] for q in gate.qubits]

            if not all(unused[min(indices) : max(indices) + 1]):
                # New layer
                lcirc = []
                layers.append(lcirc)
                unused = [True] * N

            unused[min(indices) : max(indices) + 1] = [False] * (
                max(indices) - min(indices) + 1
            )
            lcirc += gate

    # Sometimes the last layer from one moment can be merged with
    # the first layer of the next moment.
    # FIXME
    # for i in range(len(layers)-1, 1, -1):
    #     qbs0 = set(layers[i-1].qubits)
    #     qbs1 = set(layers[i].qubits)
    #     if qbs0.isdisjoint(qbs1):
    #         print(">>>>>>>>")
    #         layers[i-1].extend(layers[i])
    #         del layers[i]

    return Circuit([Circuit(L) for L in layers])


def _pretty(obj: Any, format: str = "text") -> str:
    """Pretty format an object as a text or latex string."""
    assert format in ["latex", "text"]
    if isinstance(obj, float):
        try:
            if format == "latex":
                return sympy.latex(var.asexpression(obj))
            else:
                return str(var.asexpression(obj)).replace("pi", "π")
        except ValueError:
            return f"{obj:.4g}"

    out = str(obj)
    if format == "text":
        out = out.replace("pi", "π")
    return out


# Unicode and ASCII characters for drawing boxes.
#                  t 0000000011111111
#                  r 0000111100001111
#                  b 0011001100110011
#                  l 0101010101010101
STD_BOX_CHARS = " ╴╷┐╶─┌┬╵┘│┤└┴├┼"  # noqa: E221
BOLD_BOX_CHARS = " ╸╻┓╺━┏┳╹┛┃┫┗┻┣╋"  # noqa: E221
DOUBLE_BOX_CHARS = " ═║╗══╔╦║╝║╣╚╩╠╬"  # noqa: E221  # No half widths
ASCII_BOX_CHARS = r"   \ -/+ /|+\+++"  # noqa: E221

TOP, RIGHT, BOT, LEFT = 8, 4, 2, 1
CROSS = TOP + RIGHT + BOT + LEFT


def _box_char_transpose(chars: str) -> str:
    return "".join(
        chars[utils.bitlist_to_int(list(reversed(utils.int_to_bitlist(n, 4))))]
        for n in range(16)
    )


# FIXME: pi, alpha, ect...
unicode_ascii = {
    "●": "@",
    "○": "O",
    "⁺": "^-1",
    "√": "Sqrt",
    "⟨": "<",  # "Mathematical Left Angle Bracket"
    "⟩": ">",
}
unicode_ascii.update(dict(zip(STD_BOX_CHARS, ASCII_BOX_CHARS)))


def _unicode_to_ascii(text: str) -> str:
    return "".join(unicode_ascii.get(c, c) for c in text)


# fin
