# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# CHECME copywrite, and on ops

import inspect
from typing import ClassVar, Dict, List, Type

from ..ops import _EXCLUDED_OPERATIONS, Gate

__all__ = (
    "StdGate",
    "STDGATES",
)


STDGATES: Dict[str, "Type[StdGate]"] = {}
"""All standard gates (All non-abstract subclasses of StdGate)"""


class StdGate(Gate):
    """
    A standard gate. Standard gates have a name, a fixed number of real
    parameters, and act upon a fixed number of qubits.

    e.g. Rx(theta, q0), CNot(q0, q1), Can(tx, ty, tz, q0, q1, q2)

    In the argument list, parameters are first, then qubits. Parameters
    have type Variable (either a concrete floating point number, or a symbolic
    expression), and qubits have type Qubit (Any hashable python type).
    """

    # deprecated. Use STDGATES
    cv_stdgates: ClassVar[Dict[str, Type["StdGate"]]] = {}
    """A dictionary between names and types for all StdGate subclasses"""

    def __init_subclass__(cls) -> None:
        # Note: The __init_subclass__ initializes all subclasses of a given class.
        # see https://www.python.org/dev/peps/pep-0487/

        if inspect.isabstract(cls):
            return  # pragma: no cover

        super().__init_subclass__()
        if cls.__name__ not in _EXCLUDED_OPERATIONS:
            STDGATES[cls.__name__] = cls  # Subclass registration

        # Parse the Gate arguments and number of qubits from the arguments to __init__
        # Convention is that qubit names start with "q", but arguments do not.
        names = getattr(cls, "__init__").__annotations__.keys()
        args = tuple(s for s in names if s[0] != "q" and s != "return")
        qubit_nb = len(names) - len(args)
        if "return" in names:
            # For unknown reasons, "return" is often (but not always) in names.
            qubit_nb -= 1

        cls.cv_args = args
        cls.cv_qubit_nb = qubit_nb

        # deprecated
        cls.cv_stdgates[cls.__name__] = cls  # Subclass registration

    def __repr__(self) -> str:
        args: List[str] = []
        args.extend(str(p) for p in self.params)
        args.extend(str(qubit) for qubit in self.qubits)
        fargs = ", ".join(args)

        return f"{self.name}({fargs})"


# End class StdGate
