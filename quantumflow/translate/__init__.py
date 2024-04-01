# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Translate a quantum gate into an equivalent sequence of other quantum
gates.

These translations are all analytic, so that we can use symbolic
parameters. Numerical decompositions can be found in the decompositions module.

Translations return an Iterator over gates, rather than a Circuit,
so that we can use type annotations to keep track of the source and resultant
gates of each translation.
"""

# Implementation note:
#
# Translations are all analytic, so that we can use symbolic
# parameters. Numerical decompositions can be found in decompositions.py
#
# Translations return an Iterator over gates, rather than a Circuit, in part
# so that we can use type annotations to keep track of the source and resultant
# gates of each translation.

from .translate_gates import *  # noqa: F401, F403
from .translate_stdgates_1q import *  # noqa: F401, F403
from .translate_stdgates_2q import *  # noqa: F401, F403
from .translate_stdgates_3q import *  # noqa: F401, F403
from .translations import *  # noqa: F401, F403

# fin
