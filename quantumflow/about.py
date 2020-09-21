# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# Command line interface for the about() function
# > python -m python_template.about
#
# NB: This module should not be imported by any other code in the package
# (else we will get multiple import warnings)

if __name__ == "__main__":
    import quantumflow as qf

    qf.about()
