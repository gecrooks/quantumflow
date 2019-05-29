
"""
QuantumFlow: Gradients of parameterized gates, and gradient descent optimizers
"""
from typing import Tuple, Sequence, Callable

from .circuits import Circuit
from . import backend as bk
from .measures import state_fidelity
from .qubits import asarray
from .states import State
from .ops import Operation, Gate
from .gates import join_gates

import numpy as np
from numpy import pi

from .stdgates import RX, RY, RZ, TX, TY, TZ, XX, YY, ZZ, X, Y, Z

__all__ = ('GRADIENT_GATESET',
           'expectation_gradients',
           'state_fidelity_gradients',
           'state_angle_gradients',
           'parameter_shift_circuits',
           # 'SGD',
           'Adam',
           )


GRADIENT_GATESET = frozenset(['RX', 'RY', 'RZ',
                              'TX', 'TY', 'TZ',
                              'XX', 'YY', 'ZZ'])
"""
The names of all parameterized gates for which we know how to take gradients.
"""

_UNDIFFERENTIABLE_GATE_MSG = "Undifferentiable gate"

# TODO: Move this data into gate classes?
shift_constant = {
        RX: 1/2,
        RY: 1/2,
        RZ: 1/2,
        TX: pi/2,
        TY: pi/2,
        TZ: pi/2,
        XX: pi/2,
        YY: pi/2,
        ZZ: pi/2}


gate_generator = {
        RX: X(),
        RY: Y(),
        RZ: Z(),
        TX: X(),
        TY: Y(),
        TZ: Z(),
        XX: join_gates(X(0), X(1)),
        YY: join_gates(Y(0), Y(1)),
        ZZ: join_gates(Z(0), Z(1))}


def expectation_gradients(ket0: State,
                          circ: Circuit,
                          hermitian: Operation,
                          dfunc: Callable[[float], float] = None) \
        -> Sequence[float]:
    """
    Calculate the gradients of a function of expectation for a
    parameterized quantum circuit, using the middle-out algorithm.

    Args:
        ket0: An initial state.
        circ: A circuit that acts on the initial state.
        hermitian: A Hermitian Operation for which the expectation is evaluated
        dfunc: Derivative of func. Defaults to identity.

    Returns:
        The gradient with respect to the circuits parameters.
    """
    #  res = func(<ket0|circ^H [hermitian] circ |ket0>)
    grads = []
    forward = ket0
    back = circ.run(ket0)
    back = hermitian.run(back)
    back = circ.H.run(back)

    expectation = (bk.inner(forward.tensor, back.tensor))

    for elem in circ.elements:
        assert isinstance(elem, Gate)
        back = elem.run(back)
        forward = elem.run(forward)

        if not elem.params:     # Skip gates with no parameters
            continue

        gate_type = type(elem)
        if gate_type not in shift_constant:
            raise ValueError(_UNDIFFERENTIABLE_GATE_MSG)
        r = shift_constant[gate_type]
        gen = gate_generator[gate_type].relabel(elem.qubits)

        f0 = gen.run(forward)
        g = - 2 * r * bk.imag(bk.inner(f0.tensor, back.tensor))

        if dfunc is not None:
            g = g * dfunc(expectation)

        grads.append(g)

    return grads


def state_fidelity_gradients(ket0: State,
                             ket1: State,
                             circ: Circuit) -> Sequence[float]:
    """
    Calculate the gradients of state fidelity for a parameterized quantum
    circuit, using the middle-out algorithm.

    Args:
        ket0: An initial state.
        ket1: A target state. We calculate the fidelity between this state and
        the resultant of the circuit.
        circ: A circuit that acts on ket0.
    Returns:
        The gradients of state fidelity with respect to the circuits
        parameters.
    """
    grads = []

    forward = ket0
    back = circ.H.run(ket1)
    ol = bk.inner(forward.tensor, back.tensor)

    for elem in circ.elements:
        assert isinstance(elem, Gate)
        back = elem.run(back)
        forward = elem.run(forward)

        if not elem.params:     # Skip gates with no parameters
            continue

        gate_type = type(elem)
        if gate_type not in shift_constant:
            raise ValueError(_UNDIFFERENTIABLE_GATE_MSG)
        r = shift_constant[gate_type]
        gen = gate_generator[gate_type].relabel(elem.qubits)

        f0 = gen.run(forward)
        g = - r * 2 * bk.imag(bk.inner(f0.tensor, back.tensor) * bk.conj(ol))

        grads.append(g)

    return grads


def state_angle_gradients(ket0: State,
                          ket1: State,
                          circ: Circuit) -> Sequence[float]:
    """
    Calculate the gradients of state angle for a parameterized quantum
    circuit, using the middle-out algprithm.

    Args:
        ket0: An initial state.
        ket1: A target state. We caclucate the fidelity between this state and
            the resultant of the circuit.
        circ: A circuit that acts on ket0.
    Returns:
        The gradients of the inter-state angle with respect to the circuits
        parameters.
    """
    grads = state_fidelity_gradients(ket0, ket1, circ)
    fid = state_fidelity(circ.run(ket0), ket1)
    fid = asarray(fid)
    grads = - np.asarray(grads) / (2*np.sqrt((1-fid)*fid))
    return grads


def parameter_shift_circuits(circ: Circuit,
                             index: int) -> Tuple[float, Circuit, Circuit]:
    """
    Calculate the gradients of state angle for a parameterized quantum
    circuit, using the parameter-shift rule.

    Returns the gate shift-constant, and two circuits, circ0, circ1.
    Gradients are proportional to the difference in expectation beween the two
    circuits.

    .. code-block:: python

        r, circ0, circ1 = qf.parameter_shift_circuits(circ, n)
        fid0 = qf.state_fidelity(circ0.run(ket0), ket1)
        fid1 = qf.state_fidelity(circ1.run(ket0), ket1)
        grad = r*(fid1-fid0)


    Args:
        circ: A quantum circuit with parameterized gates
        index: The index of the target gate in the quantum circuit
    Returns:
        r: The gate shift constant
        circ0: Circuit with parameter shift on target gate.
        circ1: Circuit with negative parameter shift on target gate.

    """

    elem = circ.elements[index]
    assert isinstance(elem, Gate)
    gate_type = type(elem)
    if gate_type not in shift_constant:
        raise ValueError(_UNDIFFERENTIABLE_GATE_MSG)

    r = shift_constant[gate_type]
    param = list(elem.params.values())[0]
    gate0 = gate_type(param - 0.25*pi/r, *elem.qubits)  # type: ignore
    circ0 = Circuit(circ)
    circ0.elements[index] = gate0

    gate1 = gate_type(param + 0.25*pi/r, *elem.qubits)  # type: ignore
    circ1 = Circuit(circ)
    circ1.elements[index] = gate1

    return r, circ0, circ1


# class SGD(object):
#     """Stochastic Gradient Descent optimizer"""
#     def __init__(self, learning_rate=0.001):
#         self.iterations = 0
#         self.learning_rate = learning_rate

#     def get_update(self, params, grads):
#         """ params and grads are list of numpy arrays
#         """
#         lr = self.learning_rate
#         new_params = []
#         for p, g in zip(params, grads):
#             new_params.append(p - lr * g)

#         self.iterations += 1
#         return new_params


class Adam(object):
    """Adam optimizer.

    Args:
        learning_rate: The learning rate
        beta_1: The exponential decay rate for the 1st moment estimates.
        beta_2: The exponential decay rate for the 2nd moment estimates.
        epsilon: Numerical stability fuzz factor.

    Returns:
        bool: The return value. True for success, False otherwise.

    Refs:
        - [Adam - A Method for StochasticOptimization]
            (http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7):

        self.iterations = 0
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def get_update(self, params: np.ndarray,
                   grads: np.ndarray) -> Sequence[float]:
        self.iterations += 1
        t = self.iterations
        lr = self.learning_rate
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.epsilon

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        lr_t = lr * np.sqrt(1. - beta_2**t) / (1. - beta_1**t)

        new_params = []
        for i in range(len(params)):
            p = params[i]
            g = grads[i]
            m = self.ms[i]
            v = self.vs[i]

            m_t = (beta_1 * m) + (1. - beta_1) * g
            v_t = (beta_2 * v) + (1. - beta_2) * g**2
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + epsilon)

            self.ms[i] = m_t
            self.vs[i] = v_t
            new_params.append(p_t)

        return new_params

# Fin
