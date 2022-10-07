""" Class definition for the Quantum Operator """
from cgitb import enable
import sys
import numpy as np
from typing import *
import tensorflow as tf
import math as m
import qiskit
from qiskit import Aer
from qiskit_aer import AerError
from qiskit.quantum_info.operators.predicates import (is_hermitian_matrix,
                                                      is_unitary_matrix)


# tf.config.set_visible_devices([], 'GPU')

class QuantumOperator():
  def __init__(self, 
              circuit_dimension: int = 16,
              unitary_dimension: int = 4,
              debug_log: bool = False,
              weights: tf.Tensor = None, 
              draw_circuit: bool = False, 
              show_gpu_support: bool = False,
              enable_gpu: bool = True) -> 'QuantumOperator':
    """
    Defines the Quantum Operator.

    The Quantum Operator handles the transformation from the classical
    input into the quantum circuit.

    Args:
      circuit_dimension:  It defines how many qubits the circuit should
                          use. The default is 16 (= 4^2) because it assumes
                          an input image of 4x4.

      unitary_dimension:  It defines the input qubits to the unitary
                          operator. The default is 4 ( = 2^2) because it
                          assumes two-qubit unitary transformations.

                          The weight vector dimensions are calculated with
                          this parameter. The dimensions are defined by: 
                          circuit_dimension - 1 x unitary_dimension^2
    """

    self.circuit_dimension = circuit_dimension
    self.unitary_dimension = unitary_dimension
    self.debug_log = debug_log
    self.draw_circuit = draw_circuit
    self.show_gpu_support = show_gpu_support
    self.enable_gpu = enable_gpu
  
  def feature_map(self, image: tf.Tensor) -> tf.Tensor:

    ft_map = np.zeros((image.shape[0], 2))
    for index, value in enumerate(image):
      ft_map[index][0] = np.cos(np.pi / 2 * value)
      ft_map[index][1] = np.sin(np.pi / 2 * value)

    return ft_map
  
  def hermitian_matrix(self, weights: tf.Tensor, unitary_dimension: int = 4) -> tf.Tensor:

    diag = weights[:unitary_dimension]
    complex_range = (weights.shape[0] - unitary_dimension) // 2 + unitary_dimension
    reals = weights[unitary_dimension:complex_range]
    img = weights[complex_range:]

    assert reals.shape == img.shape

    diag_matrix = np.matrix(np.diag(diag.numpy()))

    hermitian = np.matrix(np.zeros((unitary_dimension, unitary_dimension), dtype=complex))
    hermitian[np.triu_indices(unitary_dimension, 1)] = np.array([complex(a, b) for a, b in zip(reals, img)])

    hermitian = hermitian + hermitian.H + diag_matrix

    assert is_hermitian_matrix(hermitian)
    return tf.convert_to_tensor(hermitian, dtype=tf.complex128)

  def unitary_matrices(self, weights: tf.Tensor = None, unitary_dimension: int = 4) -> tf.Tensor:

    if weights == None:
      weights = self.weights

    unitaries = []
    
    for weight in weights:
      # self.hermitian_matrix(weight, unitary_dimension)
      unitary = tf.linalg.expm(1j*self.hermitian_matrix(weight, unitary_dimension))
      assert is_unitary_matrix(unitary)
      unitaries.append(unitary)
    
    U = tf.convert_to_tensor(unitaries, dtype=tf.complex128)
    
    return U
  
  def execute(self, image: tf.Tensor = None, 
                    backend: str = "aer_simulator", 
                    draw: bool = False,
                    output_format: str = "mpl",
                    filename: str = "qiskit_circuit", 
                    shots: int = 512,
                    weights: tf.Tensor = None,
                    efficient: bool = False) -> (Any or np.ndarray):

    if image is None:
      sys.exit("This function did not receive an image")
    
    if efficient:
      return self.efficient_circuit(image, backend, draw, output_format, filename, shots, weights)
    else:
      return self.normal_circuit(image, backend, draw, output_format, filename, shots, weights)


  def normal_circuit(self,
                          image: tf.Tensor = None, 
                          backend: str = "aer_simulator", 
                          draw: bool = False,
                          output_format: str = "mpl",
                          filename: str = "qiskit_circuit", 
                          shots: int = 512,
                          weights: tf.Tensor = None):

    self.weights = tf.random.normal((self.circuit_dimension - 1, self.unitary_dimension ** 2)) if weights is None else weights

    feat_map = self.feature_map(image.numpy().flatten())
    U = self.unitary_matrices().numpy()

    quantum_circuit = qiskit.QuantumCircuit(self.circuit_dimension, 1)

    for index, feature  in enumerate(feat_map):
      quantum_circuit.initialize(feature, index)
    
    index = 0
    for layer in range(int(np.log2(self.circuit_dimension))):
      for lower in range((2**layer -1 if layer > 1 else layer), self.circuit_dimension, (2**(layer+1))):
        upper = lower + layer + 1 if layer < 2 else lower + 2**layer

        quantum_circuit.unitary(U[index], [quantum_circuit.qubits[lower], quantum_circuit.qubits[upper]], f"$U_{{{index}}}$")
        index += 1
    
    quantum_circuit.measure([self.circuit_dimension - 1], [0])
    
    if self.draw_circuit or draw:
      quantum_circuit.draw(output=output_format, filename=filename)

    try:

      circuit_backend = Aer.get_backend(backend, device='GPU')
      if 'GPU' in circuit_backend.available_devices() and self.enable_gpu:
        if self.show_gpu_support:
          print(f"The backend {backend} supports GPU. We are using it!")
          self.show_gpu_support = False
        counts = qiskit.execute(quantum_circuit, circuit_backend, shots=shots).result().get_counts()
        return counts
      else:
        circuit_backend = Aer.get_backend(backend)
        counts = qiskit.execute(quantum_circuit, circuit_backend, shots=shots).result().get_counts()
        return counts
    except AerError as e:
      print(f"This module generated the following error [{e}]")

  
  def efficient_circuit(self,
                            image: tf.Tensor = None, 
                            backend: str = "aer_simulator", 
                            draw: bool = False,
                            output_format: str = "mpl",
                            filename: str = "qiskit_circuit", 
                            shots: int = 512,
                            weights: tf.Tensor = None):
      
    self.weights = tf.random.normal((self.circuit_dimension - 3, (self.unitary_dimension * 2) ** 2)) if weights is None else weights

    if self.debug_log:
      print(self.weights.shape)

    feat_map = self.feature_map(image.numpy().flatten())
    U = self.unitary_matrices(unitary_dimension=self.unitary_dimension**2).numpy()

    quantum_circuit = qiskit.QuantumCircuit(4, 1)

    for index in range(4):
      quantum_circuit.initialize(feat_map[index], index)
    
    quantum_circuit.unitary(U[0], quantum_circuit.qubits[0:4], f'$U_{0}$')

    if self.debug_log:
      print(feat_map[1:].shape)
    for index, feat in enumerate(feat_map[4:]):
      quantum_circuit.reset(3)
      quantum_circuit.initialize(feat, 3)
      quantum_circuit.unitary(U[index + 1], quantum_circuit.qubits[0:4], f"$U_{{{index + 1}}}$")
    
    quantum_circuit.measure([0], [0])

    if self.draw_circuit or draw:
      quantum_circuit.draw(output=output_format, filename=filename)
    
    try:

      circuit_backend = Aer.get_backend(backend, device='GPU')
      if self.debug_log:
        print(circuit_backend.available_devices(), end="\n\n")
      if 'GPU' in circuit_backend.available_devices() and self.enable_gpu:
        if self.show_gpu_support:
          print(f"The backend {backend} supports GPU. We are using it!\n")
          self.show_gpu_support = False
        counts = qiskit.execute(quantum_circuit, circuit_backend, shots=shots).result().get_counts()
        return counts
      else:
        if self.show_gpu_support:
          print(f"The backend {backend} supports GPU but we chose the CPU!\n")
          self.show_gpu_support = False
        circuit_backend_cpu = Aer.get_backend(backend, device='CPU')
        counts = qiskit.execute(quantum_circuit, circuit_backend_cpu, shots=shots).result().get_counts()
        return counts
    except AerError as e:
      print(f"This module generated the following error [{e}]")
    return None