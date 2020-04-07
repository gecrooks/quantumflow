"""
QuantumFlow: prototype simulator of gate-based quantum computers.
"""

# Design Note:
#
# We import the entire public interface to the top level so that users
# don't need to know the internal package layout.
# e.g. `from quantumflow import something`
# or `import quantumflow as qf; qf.something`
#
# We don't include `.utils` since those are internal utility routines.
#
# Submodules for interfacing with external resources (.xcirq, .xforest, etc.)
# are not imported at top level, since they have external dependencies that
# are not installed by default.
#
# We use a star import here (which necessitates suppressing lint messages)
# so that the public API can be specified in the individual submodules.
# Each of these submodules should define __all__.
#
# Import order defines the import hierarchy (to avoid circular imports)
# Modules should only import modules further up the list.

from quantumflow.config import version as __version__       # noqa: F401
from quantumflow.backends import backend                    # noqa: F401
from quantumflow.variables import *                         # noqa: F401, F403
from quantumflow.qubits import *                            # noqa: F401, F403
from quantumflow.states import *                            # noqa: F401, F403
from quantumflow.ops import *                               # noqa: F401, F403
from quantumflow.stdops import *                            # noqa: F401, F403
from quantumflow.gates import *                             # noqa: F401, F403
from quantumflow.channels import *                          # noqa: F401, F403
from quantumflow.paulialgebra import *                      # noqa: F401, F403
from quantumflow.circuits import *                          # noqa: F401, F403
from quantumflow.decompositions import *                    # noqa: F401, F403
from quantumflow.measures import *                          # noqa: F401, F403
from quantumflow.dagcircuit import *                        # noqa: F401, F403
from quantumflow.visualization import *                     # noqa: F401, F403
from quantumflow.gradients import *                         # noqa: F401, F403
from quantumflow.translate import *                         # noqa: F401, F403
from quantumflow.transform import *                         # noqa: F401, F403
from quantumflow.cliffords import *                         # noqa: F401, F403

# Fin GEC 2018
