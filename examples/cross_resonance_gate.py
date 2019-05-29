
import scipy
import numpy as np
import quantumflow as qf

"""
QuantumFlow Examples:
    Validate canonical decomposition of cross resonance gate.
"""


def main():
    """CLI"""
    print('Demonstrate cross resonance gate decomposition')

    s = np.random.uniform()
    b = np.random.uniform()
    c = np.random.uniform()

    print('Params: s=%s, b=%s, c=%s' % (s, b, c))
    gate1 = CR(s, b, c)
    qf.print_gate(gate1)
    coords1 = qf.canonical_coords(gate1)

    print()
    print('Canonical Coordinates:', coords1)

    circ = cr_circuit(s, b, c)
    gate2 = circ.asgate()

    print()
    print('Cross Resonance equivelent circuit')
    print(circ)

    print()
    assert qf.gates_close(gate1, gate2)

    ang = qf.gate_angle(gate1, gate2)
    print('Angle between gate and circuit:', ang)


class CR(qf.Gate):
    def __init__(self,
                 s: float,
                 b: float,
                 c: float,
                 q0: qf.Qubit = 0,
                 q1: qf.Qubit = 1):
        gen = qf.asarray(qf.join_gates(qf.X(0), qf.I(1)).asoperator())
        gen += -b * qf.asarray(qf.join_gates(qf.Z(0), qf.X(1)).asoperator())
        gen += +c * qf.asarray(qf.join_gates(qf.I(0), qf.X(1)).asoperator())
        U = scipy.linalg.expm(-0.5j * np.pi * s * gen)
        super().__init__(U, (q0, q1), dict(s=s, b=b, c=c))


def cr_circuit(s: float,
               b: float,
               c: float,
               q0: qf.Qubit = 0,
               q1: qf.Qubit = 1):

    t7 = np.arccos(((1 + b**2 * np.cos(np.pi * np.sqrt(1 + b**2) * s)))
                   / (1 + b**2)) / np.pi
    t4 = c * s
    t1 = np.arccos(np.cos(0.5*np.pi * np.sqrt(1 + b**2) * s)
                   / np.cos(t7*np.pi/2))/np.pi

    circ = qf.Circuit()
    circ += qf.TX(t1, q0)
    circ += qf.TY(1.5, q0)
    circ += qf.X(q0)
    circ += qf.TX(t4, q1)
    circ += qf.XX(t7, q0, q1)
    circ += qf.TY(1.5, q0)
    circ += qf.X(q0)
    circ += qf.TX(t1, q0)

    return circ


if __name__ == "__main__":
    main()
