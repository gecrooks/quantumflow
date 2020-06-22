
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow.xforest

Interface to pyQuil and the Rigetti Forest.

.. autoclass:: QuantumFlowQVM
    :members:

.. autofunction:: circuit_to_pyquil
.. autofunction:: pyquil_to_circuit
.. autofunction:: quil_to_program
.. autofunction:: pyquil_to_program
.. autofunction:: pyquil_to_image
.. autofunction:: wavefunction_to_state
.. autofunction:: state_to_wavefunction

.. autodata:: QUIL_GATES

"""

from typing import Sequence, Any

# import networkx as nx
import PIL
# import numpy as np

# from ..qubits import Qubit
from ..states import State
from ..gates import NAMED_GATES
from ..ops import Gate
from ..stdops import Measure
from ..circuits import Circuit
from ..stdops import Reset
from ..visualization import circuit_to_image
from .cbits import Register
from .programs import (Program, Wait, Call, Jump, Label, JumpWhen, JumpUnless,
                       Pragma, Nop, Declare, Halt, Load, Store)
from .programs import (EQ, LT, LE, GT, GE, Add, Mul, Div, Sub, And,
                       Exchange, Ior, Move, Neg, Not, Xor)
from .programs import Include, DefCircuit, Instruction       # noqa: F401
# from pyquil.api._quantum_computer import _get_qvm_compiler_based_on_endpoint

from . import pyquil

# TODO: Include more here so external users can just import xforest,
# not subpackages

__all__ = [
    # 'get_virtual_qc',
    # 'get_compiler',
    'NullCompiler',
    'circuit_to_pyquil',
    'quil_to_program',
    'pyquil_to_program',
    'pyquil_to_circuit',
    # 'qvm_run_and_measure',
    'wavefunction_to_state',
    'state_to_wavefunction',
    'QUIL_RESERVED_WORDS',
    'QuantumFlowQVM',
    'pyquil_to_image',
    'QUIL_GATES']

"""Names of Quil compatible gates"""

QUIL_RESERVED_WORDS = ['DEFGATE', 'DEFCIRCUIT', 'MEASURE', 'LABEL', 'HALT',
                       'JUMP', 'JUMP-WHEN', 'JUMP-UNLESS', 'RESET', 'WAIT',
                       'NOP', 'INCLUDE', 'PRAGMA', 'DECLARE', 'NEG', 'NOT',
                       'AND', 'IOR', 'XOR', 'MOVE', 'EXCHANGE', 'CONVERT',
                       'ADD', 'SUB', 'MUL', 'DIV', 'EQ', 'GT', 'GE', 'LT',
                       'LE', 'LOAD', 'STORE', 'TRUE', 'FALSE', 'OR']
"""Quil keywords"""


# Quil has XY but with different parameterization
# DOCME
# TODO: Should be gate classes instead of names?
QUIL_GATES = {'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'PhaseShift',
              'RX', 'RY', 'RZ', 'CZ', 'CNOT', 'SWAP',
              'ISWAP', 'CPHASE00', 'CPHASE01', 'CPHASE10',
              'CPHASE', 'PSWAP', 'CCNOT', 'CSWAP'}


class NullCompiler(pyquil.AbstractCompiler):
    """A null pyQuil compiler. Passes programs through unchanged"""
    def get_version_info(self) -> dict:
        return {}

    def quil_to_native_quil(self, program: pyquil.Program) -> pyquil.Program:
        return program

    def native_quil_to_executable(self, nq_program: pyquil.Program) \
            -> pyquil.Program:
        return nq_program


def pyquil_to_image(program: pyquil.Program) -> PIL.Image:  # pragma: no cover
    """Returns an image of a pyquil circuit.

    See circuit_to_latex() for more details.
    """
    circ = pyquil_to_circuit(program)
    img = circuit_to_image(circ)
    return img


def circuit_to_pyquil(circuit: Circuit) -> pyquil.Program:
    """Convert a QuantumFlow circuit to a pyQuil program"""
    prog = pyquil.Program()

    for elem in circuit:
        if isinstance(elem, Gate) and elem.name in QUIL_GATES:
            params = list(elem.params.values()) if elem.params else []
            prog.gate(elem.name, params, elem.qubits)
        elif isinstance(elem, Measure):
            prog.measure(elem.qubit, elem.cbit)
        else:
            # FIXME: more informative error message
            raise ValueError('Cannot convert operation to pyquil')

    return prog


def pyquil_to_circuit(program: pyquil.Program) -> Circuit:
    """Convert a protoquil pyQuil program to a QuantumFlow Circuit"""

    circ = Circuit()
    for inst in program.instructions:
        # print(type(inst))
        if isinstance(inst, pyquil.Declare):            # Ignore
            continue
        if isinstance(inst, pyquil.Halt):               # Ignore
            continue
        if isinstance(inst, pyquil.Pragma):             # TODO Barrier?
            continue
        elif isinstance(inst, pyquil.Measurement):
            circ += Measure(inst.qubit.index)
        # elif isinstance(inst, pyquil.ResetQubit):     # TODO
        #     continue
        elif isinstance(inst, pyquil.Gate):
            defgate = NAMED_GATES[inst.name]
            gate = defgate(*inst.params)
            qubits = [q.index for q in inst.qubits]
            gate = gate.relabel(qubits)
            circ += gate
        else:
            raise ValueError('PyQuil program is not protoquil')

    return circ


def quil_to_program(quil: str) -> Program:
    """Parse a quil program and return a Program object"""
    pyquil_instructions = pyquil.parser.parse(quil)
    return pyquil_to_program(pyquil_instructions)


def pyquil_to_program(program: pyquil.Program) -> Program:
    """Convert a  pyQuil program to a QuantumFlow Program"""
    def _reg(mem: Any) -> Any:
        if isinstance(mem, pyquil.MemoryReference):
            return Register(mem.name)[mem.offset]
        return mem

    prog = Program()
    for inst in program:
        if isinstance(inst, pyquil.Gate):
            qubits = [q.index for q in inst.qubits]
            if inst.name in NAMED_GATES:
                defgate = NAMED_GATES[inst.name]
                gate = defgate(*inst.params)
                gate = gate.relabel(qubits)
                prog += gate
            else:
                prog += Call(inst.name, inst.params, qubits)

        elif isinstance(inst, pyquil.ClassicalEqual):
            prog += EQ(_reg(inst.target), _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalLessThan):
            prog += LT(_reg(inst.target), _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalLessEqual):
            prog += LE(_reg(inst.target), _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalGreaterThan):
            prog += GT(_reg(inst.target), _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalGreaterEqual):
            prog += GE(_reg(inst.target), _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalAdd):
            prog += Add(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalMul):
            prog += Mul(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalDiv):
            prog += Div(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalSub):
            prog += Sub(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalAnd):
            prog += And(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalExchange):
            prog += Exchange(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalInclusiveOr):
            prog += Ior(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalMove):
            # Also handles deprecated ClassicalTrue and ClassicalFalse
            prog += Move(_reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalNeg):
            prog += Neg(_reg(inst.target))

        elif isinstance(inst, pyquil.ClassicalNot):
            prog += Not(_reg(inst.target))

        elif isinstance(inst, pyquil.ClassicalExclusiveOr):
            target = _reg(inst.left)
            source = _reg(inst.right)
            prog += Xor(target, source)

        elif isinstance(inst, pyquil.Declare):
            prog += Declare(inst.name,
                            inst.memory_type,
                            inst.memory_size,
                            inst.shared_region)

        elif isinstance(inst, pyquil.Halt):
            prog += Halt()

        elif isinstance(inst, pyquil.Jump):
            prog += Jump(inst.target.name)

        elif isinstance(inst, pyquil.JumpTarget):
            prog += Label(inst.label.name)

        elif isinstance(inst, pyquil.JumpWhen):
            prog += JumpWhen(inst.target.name, _reg(inst.condition))

        elif isinstance(inst, pyquil.JumpUnless):
            prog += JumpUnless(inst.target.name, _reg(inst.condition))

        elif isinstance(inst, pyquil.Pragma):
            prog += Pragma(inst.command, inst.args, inst.freeform_string)

        elif isinstance(inst, pyquil.Measurement):
            if inst.classical_reg is None:
                prog += Measure(inst.qubit.index)
            else:
                prog += Measure(inst.qubit.index, _reg(inst.classical_reg))

        elif isinstance(inst, pyquil.Nop):
            prog += Nop()

        elif isinstance(inst, pyquil.Wait):
            prog += Wait()

        elif isinstance(inst, pyquil.Reset):
            prog += Reset()

        elif isinstance(inst, pyquil.ResetQubit):
            prog += Reset(inst.qubit)

        elif isinstance(inst, pyquil.ClassicalStore):
            prog += Store(inst.target, _reg(inst.left), _reg(inst.right))

        elif isinstance(inst, pyquil.ClassicalLoad):
            prog += Load(_reg(inst.target), inst.left, _reg(inst.right))

        # elif isinstance(inst, pyquil.ClassicalConvert):
        #     prog += Convert(inst.target, _reg(inst.left), _reg(inst.right))

        else:  # pragma: no cover
            raise ValueError(f'Unknown pyQuil instruction: {inst}')

    return prog


def state_to_wavefunction(state: State) -> pyquil.Wavefunction:
    """Convert a QuantumFlow state to a pyQuil Wavefunction"""
    # TODO: qubits?
    amplitudes = state.vec.asarray()

    # pyQuil labels states backwards.
    amplitudes = amplitudes.transpose()
    amplitudes = amplitudes.reshape([amplitudes.size])
    return pyquil.Wavefunction(amplitudes)


def wavefunction_to_state(wfn: pyquil.Wavefunction) -> State:
    """Convert a pyQuil Wavefunction to a QuantumFlow State"""
    # TODO: qubits?
    return State(wfn.amplitudes.transpose())


# Note QAM status: 'connected', 'loaded', 'running', 'done'

class QuantumFlowQVM(pyquil.api.QAM):
    """A Quantum Virtual Machine that runs pyQuil programs."""

    def __init__(self) -> None:
        super().__init__()
        self.program: pyquil.Program
        self.status = 'connected'
        self._prog: Program
        self._ket: State

    def load(self, binary: pyquil.Program) -> 'QuantumFlowQVM':
        """
        Load a pyQuil program, and initialize QVM into a fresh state.

        Args:
            binary: A pyQuil program
        """

        assert self.status in ['connected', 'done']
        prog = quil_to_program(str(binary))

        self._prog = prog
        self.program = binary
        self.status = 'loaded'

        return self

    def write_memory(self, *, region_name: str,
                     offset: int = 0, value: int = None) -> 'QuantumFlowQVM':
        # assert self.status in ['loaded', 'done']
        raise NotImplementedError()

    def run(self) -> 'QuantumFlowQVM':
        """Run a previously loaded program"""
        assert self.status in ['loaded']
        self.status = 'running'
        self._ket = self._prog.run()
        # Should set state to 'done' after run complete.
        # Makes no sense to keep status at running. But pyQuil's
        # QuantumComputer calls wait() after run, which expects state to be
        # 'running', and whose only effect to is to set state to 'done'
        return self

    def wait(self) -> 'QuantumFlowQVM':
        assert self.status == 'running'
        self.status = 'done'
        return self

    # Note: The star is here to prevent positional arguments (apparently)
    def read_from_memory_region(self, *, region_name: str,
                                offsets: Sequence[int] = None) \
            -> Sequence[int]:
        assert self.status == 'done'

        # Unclear what offsets is meant to do. Some pyquil code seems to think
        # its synonymous with classical_addresses, but other bits claim it's
        # an unsupported weird new feature of quil's memory model. Ignore.
        if offsets is not None:
            raise NotImplementedError('Offsets not yet supported')

        reg = Register(region_name)
        assert self._ket is not None
        memory = self._ket.memory
        result = [value for addr, value in memory.items()
                  if addr.register == reg]

        return result

    def wavefunction(self) -> pyquil.Wavefunction:
        """
        Return the wavefunction of a completed program.
        """
        assert self.status == 'done'
        assert self._ket is not None
        wavefn = state_to_wavefunction(self._ket)
        return wavefn
