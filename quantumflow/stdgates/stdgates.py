# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# CHECME copywrite

# DO Rename

from typing import ClassVar, Dict, List, Mapping, Type, TypeVar

import numpy as np
import scipy

from ..config import CONJ, CTRL, SQRT
from ..future import cached_property
from ..ops import _EXCLUDED_OPERATIONS, Gate
from ..paulialgebra import Pauli, sZ
from ..qubits import Qubit, Qubits
from ..tensors import QubitTensor, asqutensor
from ..var import Variable

__all__ = (
    "StdGate",
    "STDGATES",
    "StdCtrlGate",
    "STDCTRLGATES",
)


STDGATES: Dict[str, "Type[StdGate]"] = {}
"""All standard gates (All non-abstract subclasses of StdGate)"""

STDCTRLGATES: Dict[str, "Type[StdCtrlGate]"] = {}
"""All standard control gates (All non-abstract subclasses of StdCtlGate)"""


class StdGate(Gate):
    """
    A standard gate. Standard gates have a name, a fixed number of real
    parameters, and act upon a fixed number of qubits.

    e.g. Rx(theta, q0), CNot(q0, q1), Can(tx, ty, tz, q0, q1, q2)

    In the argument list, parameters are first, then qubits. Parameters
    have type Variable (either a concrete floating point number, or a symbolic
    expression), and qubits have type Qubit (Any hashable python type).
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls.__name__ not in _EXCLUDED_OPERATIONS:
            STDGATES[cls.__name__] = cls  # Subclass registration

        # Parse the Gate arguments and number of qubits from the arguments to __init__
        args = []
        qubit_nb = 0
        for name, namet in cls.__init__.__annotations__.items():
            if namet == Variable:
                args.append(name)
            elif namet == Qubit:
                qubit_nb += 1

        cls.cv_args = tuple(args)
        cls.cv_qubit_nb = qubit_nb

    def __repr__(self) -> str:
        args: List[str] = []
        args.extend(str(p) for p in self.params)
        args.extend(str(qubit) for qubit in self.qubits)
        fargs = ", ".join(args)

        return f"{self.name}({fargs})"

    def _diagram_labels_(self) -> List[str]:
        label = self.name

        label = label.replace("ISwap", "iSwap")
        label = label.replace("Phased", "Ph")

        if label.startswith("Sqrt"):
            label = SQRT + label[4:]

        if label.endswith("_H"):
            label = label[:-2] + CONJ

        args = ""
        if self.cv_args:
            args = ", ".join("{" + arg + "}" for arg in self.cv_args)

        if args:
            if label.endswith("Pow"):
                label = label[:-3] + "^" + args
            else:
                label = label + "(" + args + ")"

        labels = [label] * self.qubit_nb

        if self.qubit_nb > 1 and not self.cv_interchangeable:
            for i in range(self.qubit_nb):
                labels[i] = labels[i] + "_%s" % i

        return labels


# End class StdGate

StdCtrlGateTV = TypeVar("StdCtrlGateTV", bound="StdCtrlGate")


class StdCtrlGate(StdGate):
    """A standard gate that is a controlled version of another standard gate.

    Subclasses should set the `cv_target` class variable to the target gate type.
    """

    # nb: ControlGate and StdCtrlGate share interface and code.
    # But unification probably not worth the trouble

    from .stdgates_1q import I

    cv_target: ClassVar[Type[StdGate]] = I  # null gate
    """StdGate type that is the target of this controlled gate.
    Should be set by subclasses"""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        assert cls.cv_target is not None
        assert cls.cv_target.cv_args == cls.cv_args

        if cls.__name__ not in _EXCLUDED_OPERATIONS:
            STDCTRLGATES[cls.__name__] = cls

    @property
    def control_qubits(self) -> Qubits:
        return self.qubits[: self.control_qubit_nb]

    @property
    def target_qubits(self) -> Qubits:
        return self.qubits[self.control_qubit_nb :]

    @property
    def control_qubit_nb(self) -> int:
        return self.qubit_nb - self.cv_target.cv_qubit_nb

    @property
    def target_qubit_nb(self) -> int:
        return self.cv_target.cv_qubit_nb

    @property
    def target(self) -> StdGate:
        return self.cv_target(*self.params, *self.target_qubits)  # type: ignore

    @property
    def hamiltonian(self) -> Pauli:
        ham = self.target.hamiltonian
        for q in self.control_qubits:
            ham *= (1 - sZ(q)) / 2
        return ham

    @cached_property
    def tensor(self) -> QubitTensor:
        ctrl_block = np.identity(2**self.cv_qubit_nb - 2**self.cv_target.cv_qubit_nb)
        target_block = self.target.asoperator()
        unitary = scipy.linalg.block_diag(ctrl_block, target_block)

        return asqutensor(unitary)

    def resolve(self: StdCtrlGateTV, subs: Mapping[str, float]) -> StdCtrlGateTV:
        target = self.target.resolve(subs)
        return type(self)(*target.params, *self.qubits)  # type: ignore

    def _diagram_labels_(self) -> List[str]:
        return ([CTRL] * self.control_qubit_nb) + self.target._diagram_labels_()


# end StdCtrlGate
