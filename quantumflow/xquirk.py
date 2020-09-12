# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow.xquirk

Interface between Quirk and QuantumFlow

https://algassert.com/quirk

.. autofunction:: circuit_to_quirk
.. autofunction:: quirk_url
.. autofunction:: open_quirk_webserver
"""
import json
import urllib
import webbrowser
from typing import Dict, List, cast

from .circuits import Circuit

__all__ = "circuit_to_quirk", "quirk_url", "open_quirk_webserver"


# TODOs
# compile to quirk.
# connectivity: non-control multi-qubit gates have to be contiguous.
#
# CCZPow
# CNotPow
# CCXPow
#
# measurement
# Iden
# Permutations:
# Rotate, Reverse, left rotate, right rotate, Interleave, Deinterleave
#
# Displays
#
# P0, P1
#
# Add QUIRK_OPS
# TODO: Docs

quirk_labels: Dict[str, List] = {
    "I": [1],
    "H": ["H"],
    "X": ["X"],
    "Y": ["Y"],
    "Z": ["Z"],
    "V": ["X^½"],
    "V_H": ["X^-½"],
    "SqrtY": ["Y^½"],
    "SqrtY_H": ["Y^-½"],
    "S": ["Z^½"],
    "S_H": ["Z^-½"],
    "T": ["Z^¼"],
    "T_H": ["Z^-¼"],
    "CNot": ["•", "X"],
    "CY": ["•", "Y"],
    "CZ": ["•", "Z"],
    "Swap": ["Swap", "Swap"],
    "CSwap": ["•", "Swap", "Swap"],
    "CCNot": ["•", "•", "X"],
    "CCZ": ["•", "•", "Z"],
}

quirk_formulaic = {
    "Rx": "Rxft",
    "Ry": "Ryft",
    "Rz": "Rzft",
    "XPow": "X^ft",
    "YPow": "Y^ft",
    "ZPow": "Z^ft",
}


def circuit_to_quirk(circ: Circuit) -> str:
    """Convert a QuantumFlow Circuit to a quirk circuit (represented as a
    JSON formatted string).
    """

    # Relabel qubits to consecutive integers
    N = circ.qubit_nb
    circ = circ.on(*range(0, N))

    columns = []
    col: List = [1] * N
    columns.append(col)

    for op in circ:
        qbs = cast(List[int], op.qubits)

        if op.name in quirk_labels:
            labels = quirk_labels[op.name]
            if "•" in col or not all(col[q] == 1 for q in qbs):
                # New column
                col = [1] * N
                columns.append(col)
            for i, q in enumerate(qbs):
                col[q] = labels[i]
        elif op.name in quirk_formulaic:
            q = qbs[0]
            if "•" in col or col[q] != 1:
                # New column
                col = [1] * N
                columns.append(col)
            (p,) = op.params
            col[q] = {"id": quirk_formulaic[op.name], "arg": str(p)}
        else:
            raise ValueError("Cannot convert Operation to Quirk")

    # Remove excess '1's  at end of column
    for col in columns:
        while len(col) > 1 and col[-1] == 1:
            col.pop()

    quirk = {"cols": columns}
    s = json.dumps(quirk, ensure_ascii=False).replace(" ", "")
    return s


def quirk_url(
    quirk: str,
    base_url: str = "https://algassert.com/quirk#circuit={}",
    escape: bool = False,
) -> str:
    """Given a quirk circuit, returns the corresponding quirk web-server URL"""
    if escape:
        quirk = urllib.parse.quote(quirk)
    return base_url.format(quirk)


def open_quirk_webserver(circ: Circuit) -> None:  # pragma: no cover
    """Translate a Circuit to quirk and open the quirk
    webapp in a browser window"""

    quirk = circuit_to_quirk(circ)
    url = quirk_url(quirk, escape=True)
    webbrowser.open(url)


# fin
