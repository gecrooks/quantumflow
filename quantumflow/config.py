# Copyright 2019-, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.

"""
Package wide configuration
"""

import platform
import re
import sys
import typing

import numpy as np

from .future import importlib_metadata

__all__ = ["__version__", "about"]


__version__ = importlib_metadata.version(__package__)  # type: ignore


quantum_dtype = np.complex128
"""The complex data type used for quantum data"""


def about(file: typing.TextIO = None) -> None:
    f"""Print information about this package.

     ``> python -m {__package__}.about``

    Args:
        file: Output stream (Defaults to stdout)
    """
    metadata = importlib_metadata.metadata(__package__)  # type: ignore
    print(f"# {metadata['Name']}", file=file)
    print(f"{metadata['Summary']}", file=file)
    print(f"{metadata['Home-page']}", file=file)

    name_width = 24
    versions = {}
    versions["platform"] = platform.platform(aliased=True)
    versions[__package__] = __version__
    versions["python"] = sys.version[0:5]

    for req in importlib_metadata.requires(__package__):  # type: ignore
        name = re.split("[; =><]", req)[0]
        try:
            versions[name] = importlib_metadata.version(name)  # type: ignore
        except Exception:  # pragma: no cover
            pass

    print(file=file)
    print("# Configuration", file=file)
    for name, vers in versions.items():
        print(name.ljust(name_width), vers, file=file)
    print(file=file)
