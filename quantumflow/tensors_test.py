# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import random

import numpy as np
import pytest
from numpy import pi

import quantumflow as qf
from quantumflow import tensors

from .config_test import REPS


def test_asqutensor() -> None:
    arr = np.zeros(shape=(256,))
    tensor = qf.asqutensor(arr)
    assert tensor.shape == (2,) * 8


def test_asqutensor_flatten() -> None:
    arr = np.zeros(shape=(256,))
    tensor = qf.asqutensor(arr)

    arr = tensors.flatten(tensor, rank=1)
    assert arr.shape == (256,)

    arr = tensors.flatten(tensor, rank=2)
    assert arr.shape == (16,) * 2

    arr = tensors.flatten(tensor, rank=4)
    assert arr.shape == (4,) * 4

    arr = tensors.flatten(tensor, rank=8)
    assert arr.shape == (2,) * 8


def test_inner_product() -> None:
    # also tested via fubini_study_angle

    for _ in range(REPS):
        theta = random.uniform(-4 * pi, +4 * pi)

        hs = tensors.inner(qf.Rx(theta, 0).tensor, qf.Rx(theta, 0).tensor)
        print(f"Rx({theta}), hilbert_schmidt = {hs}")
        assert np.isclose(hs / 2, 1.0)

        hs = tensors.inner(qf.Rz(theta, 0).tensor, qf.Rz(theta, 0).tensor)
        print(f"Rz({theta}), hilbert_schmidt = {hs}")
        assert np.isclose(hs / 2, 1.0)

        hs = tensors.inner(qf.Ry(theta, 0).tensor, qf.Ry(theta, 0).tensor)
        print(f"Ry({theta}), hilbert_schmidt = {hs}")
        assert np.isclose(hs / 2, 1.0)

        hs = tensors.inner(qf.PSwap(theta, 0, 1).tensor, qf.PSwap(theta, 0, 1).tensor)
        print(f"PSwap({theta}), hilbert_schmidt = {hs}")
        assert np.isclose(hs / 4, 1.0)

    with pytest.raises(ValueError):
        tensors.inner(qf.zero_state(0).tensor, qf.X(0).tensor)

    with pytest.raises(ValueError):
        tensors.inner(qf.CNot(0, 1).tensor, qf.X(0).tensor)


def test_tensormul_dimension_mismatch() -> None:
    """Test that tensormul raises ValueError for dimension mismatch."""
    state = qf.zero_state(3).tensor
    gate = qf.CNot(0, 1).tensor  # 2-qubit gate

    # Provide wrong number of indices (3 indices for 2-qubit gate)
    with pytest.raises(ValueError, match="Gate dimension"):
        tensors.tensormul(gate, state, (0, 1, 2))


def test_tensormul_diagonal_dimension_mismatch() -> None:
    """Test that tensormul_diagonal raises ValueError for dimension mismatch."""
    state = qf.zero_state(3).tensor
    diagonal = np.array([1, 1, 1, 1])  # 2-qubit diagonal
    diagonal = diagonal.reshape((2, 2))

    # Provide wrong number of indices (3 indices for 2-qubit diagonal)
    with pytest.raises(ValueError, match="Diagonal dimension"):
        tensors.tensormul_diagonal(diagonal, state, (0, 1, 2))


# fin
