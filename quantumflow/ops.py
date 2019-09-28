
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow


QuantumFlow supports several different quantum operations that act upon either
pure or mixed states (or both). The four main types are Gate, which represents
the action of an operator (typically unitary) on a state; Channel, which
represents the action of a superoperator on a mixed state (used for mixed
quantum-classical dynamics); Kraus, which represents a Channel as a collection
of operators; and Circuit, which is a list of other operations that act in
sequence. Circuits can contain any instance of the abstract quantum operation
superclass, Operation, including other circuits.

We consider the elemental quantum operations, such as Gate, Channel, and Kraus,
as immutable. (Although immutability is not enforced in general.)
Transformations of these operations return new copies. On the other hand the
composite operations Circuit and DAGCircuit are mutable.

The main types of Operation's are Gate, Channel, Kraus, Circuit, DAGCircuit,
and Pauli.

.. autoclass:: Operation
    :members:

"""

# NOTE: This file contains the two main types of operations on
# Quantum states, Gate's and Channel's, and an abstract superclass
# Operation. These need to be defined in the same module since they
# reference each other. The class unit tests are currently located
# separately, in test_gates.py, and test_channels.py.


from typing import Dict, Union, Any, Tuple
from copy import copy
from abc import ABC  # Abstract Base Class

import numpy as np
from scipy.linalg import fractional_matrix_power as matpow
import sympy
from sympy import Symbol

import quantumflow.backend as bk

from .qubits import Qubits, QubitVector, qubits_count_tuple, asarray
from .states import State, Density

from .utils import symbolize

__all__ = ['Operation', 'Gate', 'Channel', 'Parameter']


# DOCME. Document using symbolic paramters, and resolution
Parameter = Union[Symbol, float]
"""Type for gate parameters. Either a float, or a sympy Symbol"""


class Operation(ABC):
    """ An operation on a qubit state. An element of a quantum circuit.

    Abstract Base Class for Gate, Circuit, DAGCircuit, Channel, Kraus,
    and Pauli.
    """

    interchangeable = False     # Class attribute
    """ Is this a multi-qubit operation that is known to be invariant under
    permutations of qubits?"""

    diagonal = False     # Class attribute
    """Is the tensor know to be diagonal in the computation basis?"""

    def __init__(self,
                 qubits: Qubits = (),
                 params: Dict[str, Parameter] = None,
                 name: str = None) -> None:
        self._qubits: Qubits = tuple(qubits)
        self._params = params if params is not None else dict()
        self._name = self.__class__.__name__ if name is None else name

    @property
    def qubits(self) -> Qubits:
        """Return the qubits that this operation acts upon"""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return len(self.qubits)

    @property
    def name(self) -> str:
        """Return the name of this operation"""
        return self.__class__.__name__

    @property
    def params(self) -> Dict[str, Parameter]:
        """Return the parameters of this Operation"""
        return self._params

    def resolve(self, resolution: Dict[str, float]) -> 'Operation':
        """Resolve symbolic parameters"""
        op = copy(self)
        params = {k: float(sympy.N(v, subs=resolution))
                  for k, v in self._params.items()}
        op._params = params
        return op

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        raise NotImplementedError()

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this operation upon a mixed state"""
        raise NotImplementedError()

    def asgate(self) -> 'Gate':
        """Convert this quantum operation to a gate (if possible)"""
        raise NotImplementedError()

    def aschannel(self) -> 'Channel':
        """Convert this quantum operation to a channel (if possible)"""
        raise NotImplementedError()

    @property
    def H(self) -> 'Operation':
        """Return the Hermitian conjugate of this quantum operation.

        For unitary Gates (and Circuits composed of the same) the
        Hermitian conjugate returns the inverse Gate (or Circuit)"""
        raise NotImplementedError()

    @property
    def tensor(self) -> bk.BKTensor:
        """
        Returns the tensor representation of this operation (if possible)
        """
        raise NotImplementedError()

    # So that we can ``extend`` Circuits with Operations
    def __iter__(self) -> Any:
        yield self

    # Make Operations sortable. (So we can use Operations in opt_einsum
    # axis lables.)
    def __lt__(self, other: Any) -> bool:
        return id(self) < id(other)

# End class Operation


# TODO: Split out seperate TensorGate subclass for gates specified
# by a tensor directly.
class Gate(Operation):
    """
    A quantum logic gate. An operator that acts upon a collection of qubits.

    Attributes:
        Gate.params: Optional keyword parameters used to create this gate
        Gate.name : The name of this gate

    """
    # DOCME
    @classmethod
    def args(cls) -> Tuple[str, ...]:
        return tuple(s for s in getattr(cls, '__init__').__annotations__.keys()
                     if s[0] != 'q' and s != 'return' and s != 'tensor')

    # TODO: Fix parameter order tensor, qubits, params, name
    # Should be params, qubits, name?

    def __init__(self,
                 tensor: bk.TensorLike = None,
                 qubits: Qubits = None,
                 params: Dict[str, Parameter] = None,
                 name: str = None) -> None:
        """Create a new gate from a gate tensor or operator.

            params: Parameters used to define this gate
        """

        if tensor is not None:
            tensor = bk.astensorproduct(tensor)
            N = bk.ndim(tensor) // 2
            if qubits is None:
                qubits = range(N)
            elif len(qubits) != N:
                raise ValueError('Wrong number of qubits for tensor')

        if qubits is None:
            raise ValueError('Missing qubits')

        super().__init__(qubits, params, name)
        self._tensor = tensor

    @property
    def name(self) -> str:
        return self._name

    @property
    def tensor(self) -> bk.BKTensor:
        """Returns the tensor representation of gate operator"""
        return self._tensor

    @property
    def qubits(self) -> Qubits:
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        return len(self.qubits)

    def relabel(self, qubits: Qubits) -> 'Gate':
        """Return a copy of this Gate with new qubits"""
        qubits = tuple(qubits)
        assert len(qubits) == self.qubit_nb     # FIXME: raise exception
        gate = copy(self)
        gate._qubits = qubits
        return gate

    # Deprecate?
    @property
    def vec(self) -> QubitVector:
        return QubitVector(self.tensor, self.qubits)

    def permute(self, qubits: Qubits) -> 'Gate':
        """Permute the order of the qubits"""
        vec = self.vec.permute(qubits)
        return Gate(vec.tensor, qubits=vec.qubits)

    def asoperator(self) -> bk.BKTensor:
        """Return the gate tensor as a square array"""
        return self.vec.flatten().copy()

    def run(self, ket: State) -> State:
        """Apply the action of this gate upon a state"""
        qubits = self.qubits
        indices = [ket.qubits.index(q) for q in qubits]
        tensor = bk.tensormul(self.tensor, ket.tensor, tuple(indices),
                              diagonal=self.diagonal)
        return State(tensor, ket.qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this gate upon a density"""
        # TODO: implement without explicit channel creation? With Kraus?
        chan = self.aschannel()
        return chan.evolve(rho)

    # TODO:  function to QubitVector?
    def __pow__(self, t: float) -> 'Gate':
        """Return this gate raised to the given power."""
        # Note: This operation cannot be performed within the tensorflow or
        # torch backends in general. Subclasses of Gate may override
        # for special cases.
        N = self.qubit_nb
        matrix = asarray(self.vec.flatten())
        matrix = matpow(matrix, t)
        matrix = np.reshape(matrix, ([2]*(2*N)))
        return Gate(matrix, self.qubits)

    @property
    def H(self) -> 'Gate':
        return Gate(tensor=self.vec.H.tensor, qubits=self.qubits)

    # TODO: Refactor functionality into QubitVector
    def __matmul__(self, other: 'Gate') -> 'Gate':
        """Apply the action of this gate upon another gate

        Note that gate1 must contain all the qubits of qate0
        """
        if not isinstance(other, Gate):
            raise NotImplementedError()
        gate0 = self
        gate1 = other
        indices = (gate1.qubits.index(q) for q in gate0.qubits)
        tensor = bk.tensormul(gate0.tensor, gate1.tensor, tuple(indices))
        return Gate(tensor=tensor, qubits=gate1.qubits)

    def __str__(self) -> str:
        # Note: We don't want to eval tensor here.

        def _param_format(obj: Any) -> str:
            if isinstance(obj, float):
                try:
                    return str(symbolize(obj))
                except ValueError:
                    return "{}".format(obj)
            return str(obj)

        if self.name == 'Gate':
            return super().__repr__()

        fqubits = " "+" ".join([str(qubit) for qubit in self.qubits])

        if self.params:
            fparams = "(" + ", ".join(_param_format(p)
                                      for p in self.params.values()) + ")"
        else:
            fparams = ""

        return "{}{}{}".format(self.name, fparams, fqubits)

    def asgate(self) -> 'Gate':
        return self

    def aschannel(self) -> 'Channel':
        """Converts a Gate into a Channel"""
        N = self.qubit_nb
        R = 4

        tensor = bk.outer(self.tensor, self.H.tensor)
        tensor = bk.reshape(tensor, [2**N]*R)
        tensor = bk.transpose(tensor, [0, 3, 1, 2])

        return Channel(tensor, self.qubits)

    def su(self) -> 'Gate':
        """Convert gate tensor to the special unitary group."""
        rank = 2**self.qubit_nb
        U = asarray(self.asoperator())
        U /= np.linalg.det(U) ** (1/rank)
        return Gate(tensor=U, qubits=self.qubits)

# End class Gate


class Channel(Operation):
    """A quantum channel"""
    def __init__(self, tensor: bk.TensorLike,
                 qubits: Union[int, Qubits],
                 params: Dict[str, Parameter] = None,
                 name: str = None) -> None:
        _, qubits = qubits_count_tuple(qubits)  # FIXME NEEDED?

        self.vec = QubitVector(tensor, qubits)

        if params is not None:
            self._params = params

        if name is None:
            name = self.__class__.__name__
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def tensor(self) -> bk.BKTensor:
        """Return the tensor representation of the channel's superoperator"""
        return self.vec.tensor

    @property
    def qubits(self) -> Qubits:
        return self.vec.qubits

    @property
    def qubit_nb(self) -> int:
        return self.vec.qubit_nb

    def relabel(self, qubits: Qubits) -> 'Channel':
        """Return a copy of this channel with new qubits"""
        chan = copy(self)
        chan.vec = chan.vec.relabel(qubits)
        return chan

    def permute(self, qubits: Qubits) -> 'Channel':
        """Return a copy of this channel with qubits in new order"""
        vec = self.vec.permute(qubits)
        return Channel(vec.tensor, qubits=vec.qubits)

    @property
    def H(self) -> 'Channel':
        return Channel(tensor=self.vec.H.tensor, qubits=self.qubits)

    # TESTME
    @property
    def sharp(self) -> 'Channel':
        r"""Return the 'sharp' transpose of the superoperator.

        The transpose :math:`S^\#` switches the two covariant (bra)
        indices of the superoperator. (Which in our representation
        are the 2nd and 3rd super-indices)

        If :math:`S^\#` is Hermitian, then :math:`S` is a Hermitian-map
        (i.e. transforms Hermitian operators to Hermitian operators)

        Flattening the :math:`S^\#` superoperator to a matrix gives
        the Choi matrix representation. (See channel.choi())
        """

        N = self.qubit_nb

        tensor = self.tensor
        tensor = bk.reshape(tensor, [2**N] * 4)
        tensor = bk.transpose(tensor, (0, 2, 1, 3))
        tensor = bk.reshape(tensor, [2] * 4 * N)
        return Channel(tensor, self.qubits)

    def choi(self) -> bk.BKTensor:
        """Return the Choi matrix representation of this super
        operator"""
        # Put superop axes in the order [out_ket, in_bra, out_bra, in_ket]
        # and reshape to matrix
        N = self.qubit_nb
        return bk.reshape(self.sharp.tensor, [2**(N*2)] * 2)

    @classmethod
    def from_choi(cls,
                  tensor: bk.TensorLike,
                  qubits: Union[int, Qubits]) -> 'Channel':
        """Return a Channel from a Choi matrix"""
        return cls(tensor, qubits).sharp

    # TESTME
    # FIXME: Can't be right, same as choi?
    def chi(self) -> bk.BKTensor:
        """Return the chi (or process) matrix representation of this
        superoperator"""
        N = self.qubit_nb
        return bk.reshape(self.sharp.tensor, [2**(N*2)] * 2)

    def run(self, ket: State) -> 'State':
        raise TypeError()  # Not possible in general

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this channel upon a density"""
        N = rho.qubit_nb
        qubits = rho.qubits

        indices = list([qubits.index(q) for q in self.qubits]) + \
            list([qubits.index(q) + N for q in self.qubits])

        tensor = bk.tensormul(self.tensor, rho.tensor, tuple(indices))
        return Density(tensor, qubits, rho.memory)

    def asgate(self) -> 'Gate':
        raise TypeError()  # Not possible in general

    def aschannel(self) -> 'Channel':
        return self

    # FIXME: Maybe not needed, too special a case. Remove?
    # Or make sure can do other operations, such as neg, plus ect
    # Move functionality to QubitVector
    def __add__(self, other: Any) -> 'Channel':
        if isinstance(other, Channel):
            if not self.qubits == other.qubits:
                raise ValueError("Qubits must be identical")
            return Channel(self.tensor + other.tensor, self.qubits)
        raise NotImplementedError()  # Or return NotImplemented?

    # FIXME: Maybe not needed, too special a case. Remove?
    def __mul__(self, other: Any) -> 'Channel':
        return Channel(self.tensor*other, self.qubits)

    # DOCME
    # TODO: Refactor into QubitVector?
    def __matmul__(self, other: 'Channel') -> 'Channel':
        if not isinstance(other, Channel):
            raise NotImplementedError()
        chan0 = self
        chan1 = other
        N = chan1.qubit_nb
        qubits = chan1.qubits
        indices = list([chan1.qubits.index(q) for q in chan0.qubits]) + \
            list([chan1.qubits.index(q) + N for q in chan0.qubits])

        tensor = bk.tensormul(chan0.tensor, chan1.tensor, tuple(indices))

        return Channel(tensor, qubits)

    # TESTME
    def trace(self) -> bk.BKTensor:
        """Return the trace of this super operator"""
        return self.vec.trace()

    # TESTME
    def partial_trace(self, qubits: Qubits) -> 'Channel':
        """Return the partial trace over the specified qubits"""
        vec = self.vec.partial_trace(qubits)
        return Channel(vec.tensor, vec.qubits)

# End class Channel
