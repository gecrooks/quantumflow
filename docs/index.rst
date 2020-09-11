

QuantumFlow: A Quantum Algorithms Development Toolkit
-----------------------------------------------------


* Release:  |release| (|today|)
* Code: https://github.com/gecrooks/quantumflow
* Docs: https://quantumflow.readthedocs.io/

The core of QuantumFlow is a simulation of a gate based quantum computer, which can run  
on top of modern optimized tensor libraries (e.g. numpy or tensorflow). The 
tensorflow backend can calculate the analytic gradient of a quantum circuit
with respect to the circuit's parameters, and circuits can be optimized to perform a function
using (stochastic) gradient descent.




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
      misc
      pauli
      decomp
      info
      translate
      transform
      gradients  
      examples
      xforest
      xcirq
      xqiskit
      xquirk


.. toctree::
      :hidden:

      devnotes

      zreferences

* :ref:`devnotes`
* :ref:`genindex`



