# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import urllib

import pytest

import quantumflow as qf
from quantumflow.xquirk import circuit_to_quirk, quirk_url


def test_circuit_to_quirk() -> None:
    # 2-qubit gates

    quirk = "https://algassert.com/quirk#circuit={%22cols%22:[[1,%22X%22,%22%E2%80%A2%22],[%22%E2%80%A2%22,1,%22Z%22],[1,%22%E2%80%A2%22,%22Y%22],[%22Swap%22,1,%22Swap%22]]}"  # noqa: E501
    circ = qf.Circuit([qf.CNot(2, 1), qf.CZ(0, 2), qf.CY(1, 2), qf.Swap(0, 2)])
    print()
    print(urllib.parse.unquote(quirk))
    print(quirk_url(circuit_to_quirk(circ)))
    assert urllib.parse.unquote(quirk) == quirk_url(circuit_to_quirk(circ))

    # 3-qubit gates
    quirk = "https://algassert.com/quirk#circuit={%22cols%22:[[%22%E2%80%A2%22,%22%E2%80%A2%22,%22X%22],[%22%E2%80%A2%22,%22%E2%80%A2%22,%22Z%22],[%22%E2%80%A2%22,%22Swap%22,%22Swap%22]]}"  # noqa: E501
    circ = qf.Circuit([qf.CCNot(0, 1, 2), qf.CCZ(0, 1, 2), qf.CSwap(0, 1, 2)])
    print()
    print(urllib.parse.unquote(quirk))
    print(quirk_url(circuit_to_quirk(circ)))
    assert urllib.parse.unquote(quirk) == quirk_url(circuit_to_quirk(circ))

    test0 = "https://algassert.com/quirk#circuit={%22cols%22:[[%22Z%22,%22Y%22,%22X%22,%22H%22]]}"  # noqa: E501
    test0 = urllib.parse.unquote(test0)
    circ = qf.Circuit([qf.Z(0), qf.Y(1), qf.X(2), qf.H(3)])
    print(test0)
    print(quirk_url(circuit_to_quirk(circ)))
    assert test0 == quirk_url(circuit_to_quirk(circ))

    test_halfturns = "https://algassert.com/quirk#circuit={%22cols%22:[[%22X^%C2%BD%22,%22Y^%C2%BD%22,%22Z^%C2%BD%22],[%22X^-%C2%BD%22,%22Y^-%C2%BD%22,%22Z^-%C2%BD%22]]}"  # noqa: E501
    test_halfturns = urllib.parse.unquote(test_halfturns)
    circ = qf.Circuit(
        [qf.V(0), qf.SqrtY(1), qf.S(2), qf.V(0).H, qf.SqrtY(1).H, qf.S(2).H]
    )
    print(test_halfturns)
    print(quirk_url(circuit_to_quirk(circ)))
    assert test_halfturns == quirk_url(circuit_to_quirk(circ))

    quarter_turns = "https://algassert.com/quirk#circuit={%22cols%22:[[%22Z^%C2%BC%22],[%22Z^-%C2%BC%22]]}"  # noqa: E501
    s = urllib.parse.unquote(quarter_turns)
    circ = qf.Circuit([qf.T(0), qf.T(0).H])
    assert s == quirk_url(circuit_to_quirk(circ))

    # GHZ circuit
    quirk = "https://algassert.com/quirk#circuit={%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22],[1,%22%E2%80%A2%22,%22X%22]]}"  # noqa: E501
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1), qf.CNot(1, 2)])
    print(urllib.parse.unquote(quirk))
    print(quirk_url(circuit_to_quirk(circ)))
    assert urllib.parse.unquote(quirk) == quirk_url(circuit_to_quirk(circ))

    test_formulaic = "https://algassert.com/quirk#circuit={%22cols%22:[[{%22id%22:%22X^ft%22,%22arg%22:%220.1%22},{%22id%22:%22Y^ft%22,%22arg%22:%220.2%22},{%22id%22:%22Z^ft%22,%22arg%22:%220.3%22}],[{%22id%22:%22Rxft%22,%22arg%22:%220.4%22},{%22id%22:%22Ryft%22,%22arg%22:%220.5%22},{%22id%22:%22Rzft%22,%22arg%22:%220.6%22}]]}"  # noqa: E501
    s = urllib.parse.unquote(test_formulaic)
    circ = qf.Circuit(
        [
            qf.XPow(0.1, 0),
            qf.YPow(0.2, 1),
            qf.ZPow(0.3, 2),
            qf.Rx(0.4, 0),
            qf.Ry(0.5, 1),
            qf.Rz(0.6, 2),
        ]
    )
    assert s == quirk_url(circuit_to_quirk(circ))


def test_fail() -> None:
    with pytest.raises(ValueError):
        circ = qf.Circuit([qf.Can(0.1, 0.2, 0.3, 0, 1)])
        quirk_url(circuit_to_quirk(circ))


def test_url_escape() -> None:
    circ = qf.Circuit([qf.X(0)])
    quirk_url(circuit_to_quirk(circ), escape=True)


# fin
