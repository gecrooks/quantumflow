#!/usr/bin/env python

import timeit
import random
import quantumflow as qf
import time

from quantumflow import xcirq

GATES = 512
REPS = 8
QUBITS = 16


def benchmark_circuit(N, gate_nb, gate):
    """Create a random circuit with N qubits and given number of gates.
    Half the gates will be 1-qubit, and half 2-qubit
    """
    qubits = list(range(0, N))
    circ = qf.Circuit()
    K = gate.qubit_nb

    for _ in range(0, gate_nb):
        qbs = random.sample(qubits, K)
        circ += gate.relabel(qbs)

    return circ


def _cli():

    gates = [
        qf.I(), qf.X(), qf.Y(), qf.Z(), qf.S(), qf.T(), qf.H(),
        qf.TX(0.2), qf.TY(0.2), qf.TZ(0.2),
        qf.CNOT(), qf.CZ(), qf.SWAP(),
        qf.ISWAP(),
        qf.CCNOT(), qf.CCZ(), qf.CSWAP()]

    print()
    print('Gate     QF GOPS         Cirq GOPS')

    for gate in gates:
        circ = benchmark_circuit(QUBITS, GATES, gate)
        t = timeit.timeit(lambda: circ.run(), number=REPS,
                          timer=time.process_time)

        cq = xcirq.CirqSimulator(circ)
        t2 = timeit.timeit(lambda: cq.run(), number=REPS,
                           timer=time.process_time)

        gops = int((GATES*REPS)/t)
        gops = int((gops * 100) + 0.5) / 100.0

        gops2 = int((GATES*REPS)/t2)
        gops2 = int((gops2 * 100) + 0.5) / 100.0

        if gops/gops2 > 0.8:
            print(gate.name, '\t', gops, '\t', gops2)
        else:
            print(gate.name, '\t', gops, '\t', gops2, '\t☹️')


if __name__ == '__main__':
    _cli()
