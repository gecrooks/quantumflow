#!/usr/bin/env python

import random
import time
import timeit

import quantumflow as qf

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
        circ += gate.on(*qbs)

    return circ


def _cli():

    gates = [
        qf.I(0),
        qf.X(0),
        qf.Y(0),
        qf.Z(0),
        qf.S(0),
        qf.T(0),
        qf.H(0),
        qf.XPow(0.2, 0),
        qf.YPow(0.2, 0),
        qf.ZPow(0.2, 0),
        qf.CNot(0, 1),
        qf.CZ(0, 1),
        qf.Swap(0, 1),
        qf.ISwap(0, 1),
        qf.CCNot(0, 1, 2),
        qf.CCZ(0, 1, 2),
        qf.CSwap(0, 1, 2),
    ]

    print()
    print("Gate     QF GOPS         Cirq GOPS")

    # for n in range(4):
    #     circ = benchmark_circuit(QUBITS, GATES, qf.RandomGate([0,1]))
    #     t = timeit.timeit(lambda: circ.run(), number=REPS,
    #                       timer=time.process_time)
    #     gops = int((GATES*REPS)/t)
    #     gops = int((gops * 100) + 0.5) / 100.0
    #     print(f"gate qubits: {n}  gops:{gops}")

    for gate in gates:
        circ = benchmark_circuit(QUBITS, GATES, gate)
        t = timeit.timeit(lambda: circ.run(), number=REPS, timer=time.process_time)

        cq = qf.xcirq.CirqSimulator(circ)
        t2 = timeit.timeit(lambda: cq.run(), number=REPS, timer=time.process_time)

        gops = int((GATES * REPS) / t)
        gops = int((gops * 100) + 0.5) / 100.0

        gops2 = int((GATES * REPS) / t2)
        gops2 = int((gops2 * 100) + 0.5) / 100.0

        if gops / gops2 > 0.8:
            print(gate.name, "\t", gops, "\t", gops2)
        else:
            print(gate.name, "\t", gops, "\t", gops2, "\t☹️")


if __name__ == "__main__":
    _cli()
