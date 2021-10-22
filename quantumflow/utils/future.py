# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.

"""
Future perfect.

Conditional imports for backward compatibility.
"""

__all__ = ["importlib_metadata", "Protocol"]


try:
    # Python >= 3.8
    from importlib import metadata as importlib_metadata  # type: ignore
except ImportError:  # pragma: no cover
    # python < 3.8
    import importlib_metadata  # type: ignore  # noqa: F401


try:
    # Python >= 3.8
    from typing import Protocol  # type: ignore
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore  # noqa: F401


try:
    # Python >= 3.10
    from typing import TypeAlias  # type: ignore
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias  # type: ignore  # noqa: F401


# fin
