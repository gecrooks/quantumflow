# Copyright 2019-, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.

"""
Future perfect.

Conditional imports for backward compatibility.
"""

__all__ = ["importlib_metadata"]

try:
    from importlib import metadata as importlib_metadata  # type: ignore
except ImportError:  # pragma: no cover
    # python <= 3.8
    import importlib_metadata  # type: ignore  # noqa: F401
