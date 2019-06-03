

QuantumFlow: A Quantum Algorithms Development Toolkit
-----------------------------------------------------

* Code: https://github.com/gecrooks/quantumflow
* Docs: https://quantumflow.readthedocs.io/

The core of QuantumFlow is a simulation of a gate based quantum computer, which can run  
on top of modern optimized tensor libraries (numpy, tensorflow, or torch). The 
tensorflow backend can calculate the analytic gradient of a quantum circuit
with respect to the circuit's parameters, and circuits can be optimized to perform a function
using (stochastic) gradient descent. The torch and tensorflow backend can also accelerate the
quantum simulation using commodity classical GPUs.




.. toctree::
      :maxdepth: 3
      :caption: Contents:

      intro
      qubits
      states
      ops
      gates
      channels
      circuits
      decomp
      measures
      gradients
      programs
      forest   
      backends
      examples


.. toctree::
      :hidden:

      devnotes

* :ref:`devnotes`
* :ref:`genindex`



