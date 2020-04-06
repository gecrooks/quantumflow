#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Examples: Fit a 1-qubit gate using gradient descent,
using tensorflow 2.0
"""

import os
import tensorflow as tf

os.environ['QUANTUMFLOW_BACKEND'] = 'tensorflow'
import quantumflow as qf                    # noqa: E402
import quantumflow.backend as bk            # noqa: E402


class ZYZ(qf.Gate):
    r"""A Z-Y-Z decomposition of one-qubit rotations in the Bloch sphere
    The ZYZ decomposition of one-qubit rotations is
    .. math::
        \text{ZYZ}(t_0, t_1, t_2)
            = Z^{t_2} Y^{t_1} Z^{t_0}
    This is the unitary group on a 2-dimensional complex vector space, SU(2).
    Ref: See Barenco et al (1995) section 4 (Warning: gates are defined as
    conjugate of what we now use?), or Eq 4.11 of Nielsen and Chuang.
    Args:
        t0: Parameter of first parametric Z gate.
            Number of half turns on Block sphere.
        t1: Parameter of parametric Y gate.
        t2: Parameter of second parametric Z gate.
    """
    def __init__(self, t0: float, t1: float,
                 t2: float, q0: qf.Qubit = 0) -> None:
        super().__init__(params=dict(t0=t0, t1=t1, t2=t2), qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        t0, t1, t2 = self.params.values()
        ct0 = bk.ccast(bk.pi * t0)
        ct1 = bk.ccast(bk.pi * t1)
        ct2 = bk.ccast(bk.pi * t2)
        ct3 = 0

        unitary = [[bk.cis(ct3 - 0.5 * ct2 - 0.5 * ct0) * bk.cos(0.5 * ct1),
                    -bk.cis(ct3 - 0.5 * ct2 + 0.5 * ct0) * bk.sin(0.5 * ct1)],
                   [bk.cis(ct3 + 0.5 * ct2 - 0.5 * ct0) * bk.sin(0.5 * ct1),
                    bk.cis(ct3 + 0.5 * ct2 + 0.5 * ct0) * bk.cos(0.5 * ct1)]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> qf.Gate:
        t0, t1, t2 = self.params.values()
        return ZYZ(-t2, -t1, -t0, *self.qubits)


def fit_zyz(target_gate):
    """
    Tensorflow 2.0 example. Given an arbitrary one-qubit gate, use
    gradient descent to find corresponding parameters of a universal ZYZ
    gate.
    """

    steps = 1000

    dev = '/cpu:0'

    with tf.device(dev):
        t = tf.Variable(tf.random.normal([3]))

        def loss_fn():
            """Loss"""
            gate = ZYZ(t[0], t[1], t[2])
            ang = qf.fubini_study_angle(target_gate.vec, gate.vec)
            return ang

        opt = tf.optimizers.Adam(learning_rate=0.001)
        opt.minimize(loss_fn, var_list=[t])

        for step in range(steps):
            opt.minimize(loss_fn, var_list=[t])
            loss = loss_fn()
            print(step, loss.numpy())
            if loss < 0.01:
                break
        else:
            print("Failed to converge")

    return bk.evaluate(t)


if __name__ == "__main__":
    def main():
        """CLI"""
        print(fit_zyz.__doc__)

        print('Fitting randomly selected 1-qubit gate...')
        target = qf.random_gate(1)
        params = fit_zyz(target)
        print('Fitted parameters:', params)

    main()
