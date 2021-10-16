# Copyright 2021-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import inspect

import numpy as np
import pytest

import quantumflow as qf


def test_asarray() -> None:
    from quantumflow.base import _asarray
    from quantumflow.config import quantum_dtype

    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    vec = _asarray(arr, ndim=1)
    assert vec.ndim == 1
    assert vec.size == 16
    assert vec.dtype == quantum_dtype

    op = _asarray(arr, ndim=2)
    assert op.ndim == 2

    superop = _asarray(arr, ndim=4)
    assert superop.ndim == 4

    tensor = _asarray(arr)
    assert tensor.ndim == 4

    with pytest.raises(ValueError):
        _asarray([0, 1, 2, 3, 4])


def test_base_abstract() -> None:
    """Make sure base classes are in fact abstract"""
    assert inspect.isabstract(qf.BaseOperation)
    assert inspect.isabstract(qf.BaseGate)
    assert inspect.isabstract(qf.BaseStdGate)
    assert inspect.isabstract(qf.BaseStdCtrlGate)


def test_gate_permute() -> None:
    gate0 = qf.CNot(0, 1)

    backwards_cnot = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    gate10 = gate0.permute([1, 0])
    assert np.allclose(
        gate10.operator,
        backwards_cnot,
    )


def test_gate_matmul() -> None:
    gate0 = qf.CNot(0, 1) @ qf.CNot(0, 1)
    assert qf.almost_identity(gate0)

    gate1 = qf.CNot(0, 1) @ qf.CNot(1, 0) @ qf.CNot(0, 1)
    # TODO: Check same as swap
    gate2 = gate1 @ gate1
    assert qf.almost_identity(gate2)

    theta0 = 0.34
    theta1 = 0.11
    gate3 = qf.Rx(theta1, "a") @ qf.Rx(theta0, "a")
    assert qf.gates_close(gate3, qf.Rx(theta0 + theta1, "a"))

    gate4 = qf.I(0) @ qf.I(1)
    assert gate4.qubits == (1, 0)
    assert qf.almost_identity(gate4)


# fin
