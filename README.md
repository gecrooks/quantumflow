
# QuantumFlow: A Quantum Algorithms Development Toolkit

[![Build Status](https://travis-ci.org/gecrooks/quantumflow.svg?branch=master)](https://travis-ci.org/gecrooks/quantumflow) [![Documentation Status](https://readthedocs.org/projects/quantumflow/badge/?version=latest)](https://quantumflow.readthedocs.io/en/latest/?badge=latest)


QuantumFlow active development fork.

## Installation for development

It is easiest to install QuantumFlow's requirements using conda.
```
git clone https://github.com/gecrooks/quantumflow.git
cd quantumflow
conda install -c conda-forge --file requirements.txt
pip install -e .
```

You can also install with pip. However some of the requirements are tricky to install (notably tensorflow & cvxpy), and (probably) not everything in QuantumFlow will work correctly.
```
git clone https://github.com/gecrooks/quantumflow.git
cd quantumflow
pip install -r requirements.txt
pip install -e .
```
