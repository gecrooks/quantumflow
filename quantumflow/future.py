# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Backwards compatibility
"""

# -- Future perfect --

try:
    # python >= 3.8
    from importlib import metadata as importlib_metadata  # type: ignore
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore  # noqa: F401


try:
    # python >= 3.8
    from functools import cached_property  # type: ignore
except ImportError:  # pragma: no cover

    def cached_property(func):  # type: ignore
        """
        Method decorator for immutable properties.
        The result is cached on the first call.
        """

        def wrapper(instance):  # type: ignore
            attr = "_cached_property_" + func.__name__
            if hasattr(instance, attr):
                return getattr(instance, attr)
            result = func(instance)
            setattr(instance, attr, result)
            return result

        return property(wrapper)


# https://www.python.org/dev/peps/pep-0673/
# python < 3.11
# https://www.python.org/dev/peps/pep-0613/
# python < 3.10
# python < 3.8
from typing_extensions import Protocol  # noqa: F401
from typing_extensions import Self  # noqa: F401
from typing_extensions import TypeAlias  # noqa: F401

# fin
