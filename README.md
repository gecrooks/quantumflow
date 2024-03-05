#  QuantumFlow: A Quantum Algorithms Development Toolkit

A cross-compiler for gate based models of quantum computing

![Build Status](https://github.com/gecrooks/quantumflow-dev/workflows/Build/badge.svg) [![Documentation Status](https://readthedocs.org/projects/quantumflow/badge/?version=latest)](https://quantumflow.readthedocs.io/en/latest/?badge=latest) [![PyPi version](https://img.shields.io/pypi/v/quantumflow?color=brightgreen)](https://pypi.org/project/quantumflow/)


* [Tutorial](https://github.com/gecrooks/quantumflow-dev/tree/master/tutorial)
* [Source Code](https://github.com/gecrooks/quantumflow)
* [Issue Tracker](https://github.com/gecrooks/quantumflow-dev/issues)
* [API Documentation](https://quantumflow.readthedocs.io/)


## Installation

To install the latest stable release:
```
$ pip install quantumflow
```

In addition, install all of the external quantum libraries that QuantumFlow can interact with (such as cirq, qiskit, braket, ect.):
```
$ pip install quantumflow[ext]
```


To install the latest code from github ready for development:
```
$ git clone https://github.com/gecrooks/quantumflow.git
$ cd quantumflow
$ pip install -e .[dev]
```


