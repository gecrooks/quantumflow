#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow examples
"""

import quantumflow as qf


def prepare_w16():
    """
    Prepare a 16-qubit W state using sqrt(iswaps) and local gates,
    respecting linear topology

    """
    ket = qf.zero_state(16)
    circ = w16_circuit()
    ket = circ.run(ket)
    return ket


def w16_circuit() -> qf.Circuit:
    """
    Return a circuit that prepares the the 16-qubit W state using\
    sqrt(iswaps) and local gates, respecting linear topology
    """

    gates = [
        qf.X(7),
        qf.ISwap(7, 8) ** 0.5,
        qf.S(8),
        qf.Z(8),
        qf.Swap(7, 6),
        qf.Swap(6, 5),
        qf.Swap(5, 4),
        qf.Swap(8, 9),
        qf.Swap(9, 10),
        qf.Swap(10, 11),
        qf.ISwap(4, 3) ** 0.5,
        qf.S(3),
        qf.Z(3),
        qf.ISwap(11, 12) ** 0.5,
        qf.S(12),
        qf.Z(12),
        qf.Swap(3, 2),
        qf.Swap(4, 5),
        qf.Swap(11, 10),
        qf.Swap(12, 13),
        qf.ISwap(2, 1) ** 0.5,
        qf.S(1),
        qf.Z(1),
        qf.ISwap(5, 6) ** 0.5,
        qf.S(6),
        qf.Z(6),
        qf.ISwap(10, 9) ** 0.5,
        qf.S(9),
        qf.Z(9),
        qf.ISwap(13, 14) ** 0.5,
        qf.S(14),
        qf.Z(14),
        qf.ISwap(1, 0) ** 0.5,
        qf.S(0),
        qf.Z(0),
        qf.ISwap(2, 3) ** 0.5,
        qf.S(3),
        qf.Z(3),
        qf.ISwap(5, 4) ** 0.5,
        qf.S(4),
        qf.Z(4),
        qf.ISwap(6, 7) ** 0.5,
        qf.S(7),
        qf.Z(7),
        qf.ISwap(9, 8) ** 0.5,
        qf.S(8),
        qf.Z(8),
        qf.ISwap(10, 11) ** 0.5,
        qf.S(11),
        qf.Z(11),
        qf.ISwap(13, 12) ** 0.5,
        qf.S(12),
        qf.Z(12),
        qf.ISwap(14, 15) ** 0.5,
        qf.S(15),
        qf.Z(15),
    ]
    circ = qf.Circuit(gates)

    return circ


if __name__ == "__main__":

    def main():
        """CLI"""
        print(prepare_w16.__doc__)
        print("states           : probabilities")
        ket = prepare_w16()
        qf.print_probabilities(ket)

    main()
