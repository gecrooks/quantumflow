# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import Tuple

import numpy as np


def intlog2(n: int) -> int:
    return int(np.log2(n))


def tensormul(
    tensor0: np.ndarray, tensor1: np.ndarray, indices: Tuple[int, ...]
) -> np.ndarray:

    assert tensor0.ndim == 2
    assert tensor1.ndim == 1 or tensor1.ndim == 2

    D1 = tensor1.ndim

    R0 = intlog2(tensor1.size)
    R1 = intlog2(tensor1.size)
    K = len(indices)

    perm = list(indices) + [n for n in range(R1) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor1 = np.reshape(tensor1, [2] * R1)
    tensor1 = np.transpose(tensor1, perm)
    tensor1 = np.reshape(tensor1, [2 ** K, 2 ** (R1 - K)])

    tensor = np.matmul(tensor0, tensor1)

    tensor = np.reshape(tensor, [2] * R1)
    tensor = np.transpose(tensor, inv_perm)

    tensor = np.reshape(tensor, [2 ** (R0 // D1)] * D1)

    return tensor
