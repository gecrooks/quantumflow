# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow
.. contents:: :local:
.. currentmodule:: quantumflow

Qubits
======

States, gates, and various other methods accept a list of qubits labels upon
which the given State or Gate acts. A Qubit label can be any hashable python
object, but typically an integer or string. e.g. `[0, 1, 2]`, or
`['a', 'b', 'c']`. Note that some operations expect the qubits to be sortable,
so don't mix different incomparable data types.

"""


from typing import Hashable, Sequence

__all__ = ("Qubit", "Qubits")


Qubit = Hashable
"""
Type for qubits. Any hashable python object.
Qubits should be mutually sortable, so don't mix different types, e.g. integers
and strings.
"""


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


# fin
