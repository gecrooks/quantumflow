# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from quantumflow.future import cached_property


def test_cached_property() -> None:
    class Thing:
        def __init__(self, value: int) -> None:
            self.value = value

        @cached_property
        def plus1(self) -> int:
            return self.value + 1

        @cached_property
        def plus2(self) -> int:
            return self.value + 2

    two = Thing(2)
    assert two.plus1 == 2 + 1
    assert two.plus1 == 2 + 1
    assert two.plus2 == 2 + 2
    assert two.plus1 == 2 + 1

    ten = Thing(10)
    assert ten.plus1 == 10 + 1
    assert ten.plus1 == 10 + 1
    assert ten.plus2 == 10 + 2
    assert ten.plus2 == 10 + 2

    assert two.plus1 == 2 + 1
    assert two.plus2 == 2 + 2


# fin
