.. _devnotes:

=================
Development Notes
=================

.. contents:: :local:

Please refer to the github repository https://github.com/gecrooks/quantumflow for source code, and to submit issues and pull requests. Documentation is hosted at readthedocs https://quantumflow.readthedocs.io/ .


See the introduction for installation instructions.
The Makefile contains targets for various common development tasks::

	> make help


Conventions
###########

- nb -- Abbreviation for number
- N -- number of qubits
- theta -- gate angle (In Bloch sphere or equivalent)
- t -- Number of half turns in Block sphere (quarter cycles) or equivalent. (theta = pi * t)
- ket -- working state variable
- rho -- density variable
- chan -- channel variable
- circ -- circuit variable
- G -- Graph variable



GEC 2019
