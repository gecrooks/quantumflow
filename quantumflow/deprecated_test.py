# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import quantumflow as qf


def test_conditional_gate() -> None:
    controlled_gate = qf.conditional_gate(0, qf.X(1), qf.Y(1))

    state = qf.zero_state(2)
    state = controlled_gate.run(state)
    assert state.tensor[0, 1] == 1.0

    state = qf.X(0).run(state)
    state = controlled_gate.run(state)
    assert 1.0j * state.tensor[1, 0] == 1.0


# fin
