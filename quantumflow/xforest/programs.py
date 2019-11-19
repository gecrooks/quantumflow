
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. currentmodule:: quantumflow.xforest

A QuantumFlow Program is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_ A Program can be built
programatically by appending Instuctions, or can be parsed from code written
in Rigetti's Quantum Instruction Language (Quil). This Quantum Virtual Machine
represents a hybrid device with quantum computation and classical control flow.


In QF, Circuits contain only a list of Operations which are executed in
sequence, whereas Programs can contain non-linear control flow.


.. [1]  A Practical Quantum Instruction Set Architecture
        Robert S. Smith, Michael J. Curtis, William J. Zeng
        arXiv:1608.03355 https://arxiv.org/abs/1608.03355

.. doctest::

    import quantumflow as qf

    # Create an empty Program
    prog = qf.Program()

    # Measure qubit 0 and store result in classical bit (cbit) o
    prog += qf.Measure(0, 0)

    # Apply an X gate to qubit 0
    prog += qf.Call('X', params=[], qubits=[0])

    # Measure qubit 0 and store result in cbit 1.
    prog += qf.Measure(0, 1)

    # Compile and run program
    prog.run()

    # Contents of classical memory
    assert prog.memory == {0: 0, 1: 1}


.. autoclass:: Instruction
    :members:

.. autoclass:: Program
    :members:

.. autoclass:: DefCircuit
.. autoclass:: Wait
.. autoclass:: Nop
.. autoclass:: Halt
.. autoclass:: And
.. autoclass:: Move
.. autoclass:: Exchange
.. autoclass:: Not
.. autoclass:: Label
.. autoclass:: Jump
.. autoclass:: JumpWhen
.. autoclass:: JumpUnless
.. autoclass:: Pragma
.. autoclass:: Include
.. autoclass:: Call


"""

# Callable and State imported for typing pragmas

from typing import List, Generator, Dict, Union, Tuple, Callable
from abc import ABCMeta  # Abstract Base Class
from numbers import Number
import operator

from sympy import Symbol as Parameter

from ..ops import Operation
from ..qubits import Qubits
from ..states import zero_state, State, Density
from ..gates import NAMED_GATES
from .cbits import Addr, Register

__all__ = ['Instruction', 'Program', 'DefCircuit', 'Wait',
           'Nop', 'Halt',
           'Label', 'Jump', 'JumpWhen', 'JumpUnless',
           'Pragma', 'Include', 'Call',
           'Declare',
           # 'Convert',
           'Load', 'Store',
           'Parameter',
           # 'DefGate', 'quil_parameter', 'eval_parameter'
           'Neg', 'Not',
           'And', 'Ior', 'Xor',
           'Add', 'Mul', 'Sub', 'Div',
           'Move', 'Exchange',
           'EQ', 'LT', 'GT', 'LE', 'GE', 'NE',
           ]

# Private register used to store program state
_prog_state_ = Register('_prog_state_')
PC = _prog_state_['pc']
NAMEDGATES = _prog_state_['namedgates']
TARGETS = _prog_state_['targets']
WAIT = _prog_state_['wait']

HALTED = -1             # Program counter of finished programs


class Instruction(Operation):
    """
    An program instruction a hybrid quantum-classical program. Represents
    such operations as control flow and declarations.
    """

    def __init__(self) -> None:
        super().__init__()

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
        return self.__class__.__name__.upper()

    def quil(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.quil()

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        raise NotImplementedError()

    def evolve(self, rho: Density) -> Density:
        # For purely classical Instructions the action of run() and evolve()
        # are the same
        res = self.run(rho)
        assert isinstance(res, Density)
        return res


class Program(Instruction):
    """A Program for a hybrid quantum computer, following the Quil
    quantum instruction set architecture.
    """

    def __init__(self,
                 instructions: List[Operation] = None,
                 name: str = None,
                 params: dict = None) -> None:
        super().__init__()
        if instructions is None:
            instructions = []
        self.instructions = instructions

    def quil(self) -> str:
        items = [str(i) for i in self.instructions]
        items.append("")
        res = "\n".join(items)
        return res

    @property
    def qubits(self) -> Qubits:
        allqubits = [instr.qubits for instr in self.instructions]   # Gather
        qubits = [qubit for q in allqubits for qubit in q]          # Flatten
        qubits = list(set(qubits))                                  # Unique
        qubits.sort()                                               # Sort
        return qubits

    def __iadd__(self, other: Operation) -> 'Program':
        """Append an instruction to the end of the program"""
        self.instructions.append(other)
        return self

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, key: int) -> Operation:
        return self.instructions[key]

    def __iter__(self) -> Generator[Operation, None, None]:
        for inst in self.instructions:
            yield inst

    def _initilize(self, state: State) -> State:
        """Initialize program state. Called by program.run() and .evolve()"""

        targets = {}
        for pc, instr in enumerate(self):
            if isinstance(instr, Label):
                targets[instr.target] = pc

        state = state.store({PC: 0,
                            TARGETS: targets,
                            NAMEDGATES: NAMED_GATES.copy()})
        return state

    # FIXME: can't nest programs?
    def run(self, ket: State = None) -> State:
        """Compiles and runs a program. The optional program argument
        supplies the initial state and memory. Else qubits and classical
        bits start from zero states.
        """
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits)

        ket = self._initilize(ket)

        pc = 0
        while pc >= 0 and pc < len(self):
            instr = self.instructions[pc]
            ket = ket.store({PC: pc + 1})
            ket = instr.run(ket)
            pc = ket.memory[PC]

        return ket

    # DOCME
    # TESTME
    def evolve(self, rho: Density = None) -> Density:
        if rho is None:
            rho = zero_state(self.qubits).asdensity()

        rho1 = self._initilize(rho)
        assert isinstance(rho1, Density)     # Make type checker happy
        rho = rho1

        pc = 0
        while pc >= 0 and pc < len(self):
            instr = self.instructions[pc]
            rho = rho.store({PC: pc + 1})
            rho = instr.evolve(rho)
            pc = rho.memory[PC]

        return rho


# TODO: Rename DefProgram? SubProgram?, Subroutine???
# FIXME: Not implemented
class DefCircuit(Program):
    """Define a parameterized sub-circuit of instructions."""

    # Note that Quil's DEFCIRCUIT can include arbitrary instructions,
    # whereas QuantumFlows Circuit contains only quantum gates. (Called
    # protoquil in pyQuil)

    def __init__(self,
                 name: str,
                 params: Dict[str, float],
                 qubits: Qubits = None,
                 instructions: List[Operation] = None) \
            -> None:
        # DOCME: Not clear how params is meant to work
        super().__init__(instructions)
        if qubits is None:
            qubits = []
        self.progname = name
        self._params = params
        self._qubits = qubits

    @property
    def qubits(self) -> Qubits:
        return self._qubits

    def quil(self) -> str:
        if self.params:
            fparams = '(' + ','.join(map(str, self.params)) + ')'
        else:
            fparams = ""

        if self.qubits:
            # FIXME: Not clear what type self.qubits is expected to be?
            # These are named qubits?
            # Fix, test, and remove pragma
            fqubits = ' ' + ' '.join(map(str, self.qubits))  # pragma: no cover
        else:
            fqubits = ''

        result = f'{self.name} {self.progname}{fparams}{fqubits}:\n'

        for instr in self.instructions:
            result += "    "
            result += str(instr)
            result += "\n"

        return result


class Wait(Instruction):
    """Returns control to higher level by calling Program.wait()"""
    def run(self, ket: State) -> State:
        # FIXME: callback
        return ket


class Nop(Instruction):
    """No operation"""
    def run(self, ket: State) -> State:
        return ket


class Halt(Instruction):
    """Halt program and exit"""
    def run(self, ket: State) -> State:
        ket = ket.store({PC: HALTED})
        return ket


class Load(Instruction):
    """ The LOAD instruction."""

    def __init__(self, target: Addr, left: str, right: Addr) -> None:
        super().__init__()
        self.target = target
        self.left = left
        self.right = right

    def quil(self) -> str:
        return f'{self.name} {self.target} {self.left} {self.right}'

    def run(self, ket: State) -> State:
        raise NotImplementedError()


class Store(Instruction):
    """ The STORE instruction."""

    def __init__(self, target: str, left: Addr,
                 right: Union[int, Addr]) -> None:
        super().__init__()

        self.target = target
        self.left = left
        self.right = right

    def quil(self) -> str:
        return f'{self.name} {self.target} {self.left} {self.right}'

    def run(self, ket: State) -> State:
        raise NotImplementedError()


class Label(Instruction):
    """Set a jump target."""
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def quil(self) -> str:
        return f'{self.name} @{self.target}'

    def run(self, ket: State) -> State:
        return ket


class Jump(Instruction):
    """Unconditional jump to target label"""
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def quil(self) -> str:
        return f'{self.name} @{self.target}'

    def run(self, ket: State) -> State:
        dest = ket.memory[TARGETS][self.target]
        return ket.store({PC: dest})


class JumpWhen(Instruction):
    """Jump to target label if a classical bit is one."""
    def __init__(self, target: str, condition: Addr) -> None:
        super().__init__()
        self.target = target
        self.condition = condition

    def quil(self) -> str:
        return f'{self.name} @{self.target} {self.condition}'

    def run(self, ket: State) -> State:
        if ket.memory[self.condition]:
            dest = ket.memory[TARGETS][self.target]
            return ket.store({PC: dest})
        return ket

    @property
    def name(self) -> str:
        return "JUMP-WHEN"


class JumpUnless(Instruction):
    """Jump to target label if a classical bit is zero."""
    def __init__(self, target: str, condition: Addr) -> None:
        super().__init__()
        self.target = target
        self.condition = condition

    def quil(self) -> str:
        return f'{self.name} @{self.target} {self.condition}'

    def run(self, ket: State) -> State:
        if not ket.memory[self.condition]:  # pragma: no cover  # FIXME
            dest = ket.memory[TARGETS][self.target]
            return ket.store({PC: dest})
        return ket

    @property
    def name(self) -> str:
        return "JUMP-UNLESS"


class Pragma(Instruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as::
        PRAGMA <command> <arg>* "<freeform>"?
    """
    def __init__(self,
                 command: str,
                 args: List[float] = None,
                 freeform: str = None) -> None:
        super().__init__()
        self.command = command
        self.args = args
        self.freeform = freeform

    def quil(self) -> str:
        ret = [f'PRAGMA {self.command}']
        if self.args:
            ret.extend(str(a) for a in self.args)
        if self.freeform:
            ret.append(f'"{self.freeform}"')
        return ' '.join(ret)

    def run(self, ket: State) -> State:
        return ket


class Include(Instruction):
    """Include additional file of quil instructions.

    (Currently recorded, but not acted upon)
    """
    def __init__(self, filename: str, program: Program = None) -> None:
        # DOCME: What is program argument for? How is this meant to work?
        super().__init__()
        self.filename = filename
        self.program = program

    def quil(self) -> str:
        return f'{self.name} "{self.filename}"'

    def run(self, ket: State) -> State:
        raise NotImplementedError()


class Call(Instruction):
    """Pass control to a named gate or circuit"""
    def __init__(self,
                 name: str,
                 params: List[Parameter],
                 qubits: Qubits) -> None:
        super().__init__()
        self.gatename = name
        self.call_params = params
        self._qubits = qubits

    def quil(self) -> str:
        if self.qubits:
            fqubits = " "+" ".join([str(qubit) for qubit in self.qubits])
        else:
            fqubits = ""
        if self.call_params:
            fparams = "(" + ", ".join(str(p) for p in self.call_params) \
                + ")"
        else:
            fparams = ""
        return f"{self.gatename}{fparams}{fqubits}"

    def run(self, ket: State) -> State:
        namedgates = ket.memory[NAMEDGATES]
        if self.gatename not in namedgates:
            raise RuntimeError('Unknown named gate')

        gateclass = namedgates[self.gatename]
        gate = gateclass(*self.call_params)
        gate = gate.relabel(self.qubits)

        ket = gate.run(ket)
        return ket


class Declare(Instruction):
    """Declare classical memory"""
    def __init__(self, memory_name: str,
                 memory_type: str,
                 memory_size: int,
                 shared_region: str = None,
                 offsets: List[Tuple[int, str]] = None) -> None:
        super().__init__()
        self.memory_name = memory_name
        self.memory_type = memory_type
        self.memory_size = memory_size
        self.shared_region = shared_region
        self.offsets = offsets

    def quil(self) -> str:
        ql = ['DECLARE']
        ql += [self.memory_name]
        ql += [self.memory_type]
        if self.memory_size != 1:
            ql += [f'[{self.memory_size}]']

        if self.shared_region is not None:
            ql += ['SHARING']
            ql += [self.shared_region]

            # if self.offsets:
            #     for loc, name in self.offsets:
            #         ql += ['OFFSET', str(loc), name]

        return ' '.join(ql)

    def run(self, ket: State) -> State:
        reg = Register(self.memory_name, self.memory_type)
        mem = {reg[idx]: 0 for idx in range(self.memory_size)}
        return ket.store(mem)


class Neg(Operation):
    """Negate value stored in classical memory."""
    def __init__(self, target: Addr) -> None:
        super().__init__()
        self.target = target
        self.addresses = [target]

    def run(self, ket: State) -> State:
        return ket.store({self.target: - ket.memory[self.target]})

    def __str__(self) -> str:
        return f'{self.name.upper()} {self.target}'


class Not(Operation):
    """Take logical Not of a classical bit."""
    def __init__(self, target: Addr) -> None:
        super().__init__()
        self.target = target

    def run(self, ket: State) -> State:
        res = int(not ket.memory[self.target])
        return ket.store({self.target: res})

    def __str__(self) -> str:
        return f'{self.name.upper()} {self.target}'


# Binary classical operations

class BinaryOP(Operation, metaclass=ABCMeta):
    _op: Callable

    """Abstract Base Class for operations between two classical addresses"""
    def __init__(self, target: Addr, source: Union[Addr, Number]) -> None:
        super().__init__()
        self.target = target
        self.source = source

    def _source(self, state: State) -> Union[Addr, Number]:
        if isinstance(self.source, Addr):
            return state.memory[self.source]
        return self.source

    def __str__(self) -> str:
        return f'{self.name.upper()} {self.target} {self.source}'

    def run(self, ket: State) -> State:
        target = ket.memory[self.target]
        if isinstance(self.source, Addr):
            source = ket.memory[self.source]
        else:
            source = self.source

        res = self._op(target, source)
        ket = ket.store({self.target: res})
        return ket

    def evolve(self, rho: Density) -> Density:
        res = self.run(rho)
        assert isinstance(res, Density)  # Make type checker happy
        return res


class And(BinaryOP):
    """Classical logical And of two addresses. Result placed in target"""
    _op = operator.and_


class Ior(BinaryOP):
    """Take logical inclusive-or of two classical bits, and place result
    in first bit."""
    _op = operator.or_


# class Or(BinaryOP):
#     """Take logical inclusive-or of two classical bits, and place result
#     in first bit. (Deprecated in quil. Use Ior instead."""
#     _op = operator.or_


class Xor(BinaryOP):
    """Take logical exclusive-or of two classical bits, and place result
    in first bit.
    """
    _op = operator.xor


class Add(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.add


class Sub(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.sub


class Mul(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.mul


class Div(BinaryOP):
    """Add two classical values, and place result in target."""
    _op = operator.truediv


class Move(BinaryOP):
    """Copy left classical bit to right classical bit"""
    def run(self, ket: State) -> State:
        return ket.store({self.target: self._source(ket)})


class Exchange(BinaryOP):
    """Exchange two classical bits"""
    def run(self, ket: State) -> State:
        assert isinstance(self.source, Addr)
        return ket.store({self.target: ket.memory[self.source],
                          self.source: ket.memory[self.target]})


# Comparisons

class Comparison(Operation, metaclass=ABCMeta):
    """Abstract Base Class for classical comparisons"""
    _op: Callable

    def __init__(self, target: Addr, left: Addr, right: Addr) -> None:
        super().__init__()
        self.target = target
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f'{self.name} {self.target} {self.left} {self.right}'

    def run(self, ket: State) -> State:
        res = self._op(ket.memory[self.left], ket.memory[self.right])
        ket = ket.store({self.target: res})
        return ket


class EQ(Comparison):
    """Set target to boolean (left==right)"""
    _op = operator.eq


class GT(Comparison):
    """Set target to boolean (left>right)"""
    _op = operator.gt


class GE(Comparison):
    """Set target to boolean (left>=right)"""
    _op = operator.ge


class LT(Comparison):
    """Set target to boolean (left<right)"""
    _op = operator.lt


class LE(Comparison):
    """Set target to boolean (left<=right)"""
    _op = operator.le


class NE(Comparison):
    """Set target to boolean (left!=right)"""
    _op = operator.ne
