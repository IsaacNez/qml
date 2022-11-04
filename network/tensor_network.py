""" Class definition for the Tensor Network """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import operator
from typing import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.dataloader import Dataset
from network.quantum import QuantumOperator
import utils.utils as tl
from joblib import Parallel, delayed
from itertools import cycle
import gc
import pickle
import time

devices = tf.config.list_physical_devices()
for device in devices:
  if device.device_type == 'GPU':
    tf.config.experimental.set_memory_growth(device, True)


class Network():
  def __init__(self,  image_size: int = 4,
                      circuit_dim: int = 16,
                      unitary_dim: int = 4,
                      shots: int = 512,
                      param_a: float = 28.0,
                      param_b: float = 33.0,
                      param_A: float = 74.1,
                      param_s: float = 4.13,
                      param_t: float = 0.658,
                      param_lambda: float = 0.234,
                      param_eta: float = 5.59,
                      param_gamma: float =  0.882,
                      enable_transformations: bool = False,
                      enable_log: bool = False,
                      draw_circuits: bool = False,
                      epochs: int = 10,
                      batch: int = 222,
                      classes: dict = {9: "0", 0: "1"},
                      efficient: bool = False,
                      circuit_type: str = 'normal',
                      shuffle: bool = False,
                      perf_metrics: bool = False,
                      samples: int = -1) -> 'Network':
    """ Creates the Network and defines the required parameters 

    Args:
      image_size:                     It specifies the size of the image. The resulting image will be
                                      of size (image_size, image_size)

      circuit_dim:                    It specifies the number of qubits to use. A good way to define is
                                      image_size*image_size

      unitary_dim:                    It specifies the dimension of the unitary matrices. By default 
                                      (two-qubit), it is of size (unitary_dim, unitary_dim).

      shots:                          It determines how many times the experiment is run.

      param_a:                        It defines the value of a used to generate the series alpha_k

      param_b:                        It defines the value of b used to generate the series beta_k

      param_A:                        It defines the value of A used to generate the series alpha_k

      param_s:                        It defines the value of s used to generate the series alpha_k

      param_t:                        It defines the value of t used to generate the series beta_k

      param_lambda:                   It defines the value of lambda used to calculate the Loss

      param_eta:                      It defines the value of eta used to calculate the Loss

      param_gamma:                    It defines the value of gamma used for the momentum update [DEPRECRATED]

      enable_transformations:         It enables Data Augmentation for the train dataset

      enable_log:                     It enables debug outputs for this class and subclasses.

      draw_circuits:                  It indicates to the QuantumOperator if the circuit should be drawn.

      epochs:                         It defines for how many epochs the training process should run.

      batch:                          It defines the size of batch.

      classes:                        It defines the classes for binary classification. It should be of the form
                                      {digit: "<class>", digit: "<class>"}, where digit is an integer from 0-9 and
                                      <class> is 0 or 1.

      circuit_type:                   It defines the method to run the QuantumOperator. For more, read the documentation
                                      of QuantumOperator.execute

      shuffle:                        It activates the shuffling behavior for the Dataset class.

      perf_metrics:                   It output performance metrics in each batch.

      samples:                        It defines how many samples the Dataset class should generate. For more on this 
                                      parameter, read the documentation for the Dataset class.
    """
    self.image_size         = image_size
    self.circuit_dim        = circuit_dim
    self.unitary_dim        = unitary_dim
    self.shots              = shots
    self.param_a            = param_a
    self.param_b            = param_b
    self.param_A            = param_A
    self.param_s            = param_s
    self.param_t            = param_t
    self.param_lambda       = param_lambda
    self.param_eta          = param_eta
    self.param_gamma        = param_gamma
    self.enable_transform   = enable_transformations
    self.enable_log         = enable_log
    self.draw_circuit       = draw_circuits
    self.epochs             = epochs
    self.batch              = batch
    self.circuit_type       = circuit_type
    self.classes            = classes
    self.efficient          = efficient
    self.shuffle            = shuffle
    self.perf_metrics       = perf_metrics


    if self.circuit_type == 'normal' or self.circuit_type == 'experimental':
      self.shape = (self.circuit_dim - 1, self.unitary_dim ** 2)
    elif self.circuit_type == 'efficient':
      self.shape = (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2)
    else:
      raise ValueError(f"circuit_type can only be normal, efficient or experimental. We received: {circuit_type}")

    self.weights = tf.random.uniform(self.shape)

    if self.circuit_type == 'normal' or self.circuit_type == 'experimental':
      self.shape = (self.circuit_dim - 1, self.unitary_dim ** 2)
    elif self.circuit_type == 'efficient':
      self.shape = (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2)
    else:
      raise ValueError(f"circuit_type can only be normal, efficient or experimental. We received: {circuit_type}")

    self.weights = tf.random.uniform(self.shape)

    print("Welcome to TN on Qiskit!!!")
    if self.enable_log:
      print(f"\u2192 This code will be run on an image size of {self.image_size}x{self.image_size} on Qiskit with {self.shots} shots for the {self.circuit_type} circuit")
      print(f"\u2192 We will train for {self.epochs} epochs with a batch size of {self.batch} on", "a shuffled" if self.shuffle else "an unshuffled", "dataset")
    print("")

    self.dataset = Dataset( image_size=self.image_size,
                            enable_transformations=self.enable_transform,
                            enable_log=self.enable_log, filter=True, filter_by=self.classes, samples=samples, shuffle=self.shuffle)

    self.qcircuit = QuantumOperator(circuit_dimension=self.circuit_dim, draw_circuit=draw_circuits, show_gpu_support=False, enable_gpu=False)
  
  def set_quantum_operator(self, quantum_op: QuantumOperator = None) -> None:
    if quantum_op is None:
      raise ValueError("The new Quantum Operator Class cannot be empty")
    
    self.qcircuit = quantum_op

  def get_quantum_operator(self) -> QuantumOperator:
    return self.qcircuit

  def loss(self, prediction: dict, label: str, classes: dict) -> Tuple[float, int]:
    """ Loss Function per image 
    
    Args:
      prediction:         Dictionary with the predicted results.

      label:              Correct label

      classes:            Mapping of labels to classes

    Output
      Tuple[float, int]:  Tuple representing the loss and if it was a correct prediction (1) or not (0)
    """
    p_result = max(prediction.values()) / self.shots
    p_label = prediction.pop(classes[label], None) / self.shots
    p_max_false = max(prediction.values()) / self.shots
    return max(p_max_false - p_label + self.param_lambda, 0) ** self.param_eta, 1 if p_result == p_label else 0

  def execute(self, image: tf.Tensor, label: str, weights: tf.Tensor, circuit_type: str, classes: dict, device: Any or str) -> Tuple[float, int]:
    """ Executes and calculates the loss of the current execution results 
    
    Args:
      image:                Image to classify of the size (image_size, image_size)

      label:                Label of the corresponding image

      weights:              Corresponding weights of the classification model. The size should be 
                            (circuit_size-1, unitary_dimension^2) or (circuit_size-3, (unitary_dimension^2)^2)
                            depending on the circuit_type. 

      circuit_type:         Circuit type to run the classification task. Read the documentation from QuantumOperator.execute
                            for more information.

      classes:              Dictionary with mapping of digits to classes.

      device:               Device to run the Network and QuantumOperator. This value will be ignored if the system does not
                            support Hardware Acceleration

    Output:
      Tuple[float, int]:    Tuple representing the loss and if it was a correct prediction (1) or not (0)
    """
    result = self.qcircuit.execute(image, weights=weights, device=device, shots=self.shots, circuit_type=circuit_type)
    loss, correct = self.loss(result, label, classes)
    gc.collect()
    return loss, correct

  def spsa_loss(self, batch: Tuple[tf.Tensor, np.ndarray], weights: tf.Tensor, classes: dict, verbose: bool = False, name: str = "SPSA Loss") -> Tuple[float, int]:
    """ Calculates the SPSA Loss for a specific batch 

        This method uses hybrid parallelism to speedup the computation of the loss 
        per batch. If the system supports Hardware Acceleration, it will spawn threads
        (when possible) on it. By default it uses the totality of your logical cores, thus
        be careful with the size of your batch.

        Likewise, for optimal performance, try to avoid other high-intensity programs while 
        this function is running.

    Args:
      batch:                Batch of images with their corresponding label to classify. The size should be
                            (batch, image_size, image_size)

      weights:              Corresponding weights of the classification model. The size should be 
                            (circuit_size-1, unitary_dimension^2) or (circuit_size-3, (unitary_dimension^2)^2)
                            depending on the circuit_type.

      classes:              Dictionary with mapping of digits to classes.

      verbose:              If the method should output verbose logging.

      name:                 Name identifies if this function is called multiple times

    Output:
      Tuple[float, int]:    Loss and count of correct predictions.
    """
    idx_data, idx_label = batch
    total_loss, total_correct = 0.0, 0
    
    num_cpus = os.cpu_count()
    self.jobs = num_cpus if num_cpus < batch[0].shape[0] else batch[0].shape[0]
    
    if len(devices) > 1:
      self.jobs += len(devices) - 1
    results = Parallel(n_jobs=int(self.jobs), backend='loky')(delayed(self.execute)(image, label, weights, self.circuit_type, classes, device) for image, label, device in zip(idx_data, idx_label, cycle(devices)))

    for values in results:
      total_loss += values[0]
      total_correct += values[1]

    if verbose:
      print(f"The {name} is {total_loss / batch[0].shape[0]} with accuracy {total_correct / batch[0].shape[0]} ")

    batch_size = batch[0].shape[0]
    del results
    del weights
    del batch
    gc.collect()

    return (total_loss / batch_size), total_correct

  def predict(self) -> None:
    """ Calculate the prediction accuracy for the Network

      Based on the test dataset and the fitted Network parameters (weights),
      calculate the Network accuracy to predict the correct class for the images 
      in the test dataset.

      This method use hybrid parallelism where it will use Hardware Acceleration when 
      possible.

      Be careful to run other heavy-load applications alongside this method to not decrease
      its performance.
    """
    if self.enable_log:
      print("Starting testing...")

    gc.collect()

    jobs = os.cpu_count()
    
    if len(devices) > 1:
      jobs += len(devices) - 1

    results = Parallel(n_jobs=int(self.jobs))(delayed(self.qcircuit.execute)(image=sample[0], weights=self.weights, device=device, shots=self.shots, circuit_type=self.circuit_type) for sample, device in zip(self.dataset.get_test_samples(), cycle(devices)))
    
    prediction_correct = 0

    for prediction, sample in zip(results, self.dataset.get_test_samples()):
      prediction_class = max(prediction.items(), key=operator.itemgetter(1))[0]
      prediction_label = list(self.classes.keys())[list(self.classes.values()).index(prediction_class)]
      correct_label = sample[1]
      print(f"<{prediction}>::Pred. class=[{prediction_class}], Correct Label=[{correct_label}], Pred. Label=[{prediction_label}]", end="")
      if prediction_label == correct_label:
        print(" \u2714")
        prediction_correct += 1
      else:
        print(" \u2716")
    print(f"The accuracy of the model is {prediction_correct}/{self.dataset.get_test_dataset_size()}={prediction_correct/self.dataset.get_test_dataset_size()}")

  def save_model(self, path: str = 'model/',filename: str = 'tn_model.pickle') -> None:
    """ Saves the current Network to be further used 
    
    Args:
      path:         Path where the model will be saved. Before saving,
                    it will be converted into an absolute path and check
                    if it exists. When it does not exist, it will be created.

      filename:     Filename for the model. It must have the .pickle extension.
                    The filename gets appended to the absolute path before opening
                    the file descriptor.
    """
    if not os.path.exists(path):
      os.makedirs(path)
    
    output_path = os.path.abspath(path)

    with open(os.path.join(output_path, filename), 'wb+') as file:
      pickle.dump(self, file)

  @staticmethod
  def load_model(path: str = 'model/', filename: str = 'tn_model.pickle') -> 'Network':
    """ Load a model of type Network for further processing 
    
    Args:
      path:         Path where the model is located. Before opening, the existance
                    of the file is checked.

      filename:     Filename for the model. It must have the .pickle extension.
                    The filename gets appended to the absolute path before opening
                    the file descriptor.
    
    Output:
      Network:      A class object describing the Network class for further processing.

    Raised:
      IOError:      If the file on the given path does not exist, it will raise a IOError
    """
    file_path = os.path.join(os.path.abspath(path), filename)
    if not os.path.isfile(file_path):
      raise IOError(f"File {filename} cannot be found in {os.path.abspath(path)}")
    
    with open(file_path, 'rb') as file:
      model = pickle.load(file)
    return model

  def train(self, output_results: bool = False) -> None:
    """ Train the Network for the MNIST Dataset

    Args:
      output_results:     Indicate if there should be images created during
                          training.
    """
    spsa_v = tf.zeros(self.shape)
    
    self.accuracy = []
    self.loss_g = []
    self.loss_l1 = []
    self.loss_l2 = []
    self.correct_l1 = []
    self.correct_l2 = []

    mod_batch = self.dataset.get_dataset_size() % self.batch
    total_batches = self.dataset.get_dataset_size() // self.batch + (1 if mod_batch != 0 else 0)

    save_idxs = list(range(0, total_batches, int(0.1*total_batches)))
    save_idxs.append(total_batches-1)
    for epoch in range(self.epochs + 1):
      alpha_k = self.param_a / (epoch + self.param_A + 1) ** self.param_s
      beta_k  = self.param_b / (epoch + 1) ** self.param_t

      epoch_correct = 0

      if self.enable_log:
        print(f"Epoch {epoch}/{self.epochs}")

      num_batch = 0
      for batch in self.dataset.get_batch(batch=self.batch):
        print(f"Batch: {num_batch}")
        # b = self.param_b
        # mean = tf.math.reduce_mean(batch[0]) / (2*b)
        # a = self.param_a/mean
        # alpha_k = a / (epoch + mean + 1) ** self.param_s
        # beta_k  = b / (epoch + 1) ** self.param_t

        delta = tfp.distributions.Binomial(batch[0].shape[0], probs=0.5).sample(sample_shape=self.shape)

        weights_neg, weights_pos = self.weights - alpha_k * delta, self.weights + alpha_k * delta
        if self.perf_metrics:
          start_batch = time.time()
        
        l_tilde_1, correct = self.spsa_loss(batch, weights_pos, self.classes, True, "L1 Loss")
        l_tilde_2, correct_2 = self.spsa_loss(batch, weights_neg, self.classes, True, "L2 Loss")
        
        if self.perf_metrics:
          print(f"Total batch time is: {time.time() - start_batch} seconds")  

        g = (l_tilde_1 - l_tilde_2) / (2*alpha_k)
        
        self.loss_g.append(g)
        self.loss_l1.append(l_tilde_1)
        self.loss_l2.append(l_tilde_2)
        self.correct_l1.append(correct / batch[0].shape[0])
        self.correct_l2.append(correct_2 / batch[0].shape[0])
        
        # spsa_v = self.param_gamma * spsa_v - g * beta_k * delta

        # self.weights = self.weights + spsa_v
        self.weights -= g *beta_k* delta

        
        if output_results and num_batch in save_idxs:
          tl.generate_plot(y_value=self.loss_g, x_label="Total Batches", y_label="Modified SPSA Loss", title="Training Convergence", save_plot=True, filename=f"circuit_{self.circuit_type}_loss_{self.image_size}x{self.image_size}.png", marker='.')
          tl.generate_plot(y_value=self.loss_l1, x_label="Total Batches", y_label=r"SPSA Loss $L(\Lambda+\alpha\Delta$)", title="Total Loss", save_plot=True, filename=f"circuit_{self.circuit_type}_total_loss_L1_{self.image_size}x{self.image_size}.png", marker='.')
          tl.generate_plot(y_value=self.loss_l2, x_label="Total Batches", y_label=r"SPSA Loss $L(\Lambda-\alpha\Delta$)", title="Total Loss", save_plot=True, filename=f"circuit_{self.circuit_type}_total_loss_L2_{self.image_size}x{self.image_size}.png", marker='.')
          tl.generate_plot(y_value=self.correct_l1, x_label="Total Batches", y_label="Accuracy (%)", title=r"Accuracy for $L(\Lambda+\alpha\Delta$)", save_plot=True, filename=f"circuit_{self.circuit_type}_accuracy_L1_{self.image_size}x{self.image_size}.png", ylim=[0,1], marker='.')
          tl.generate_plot(y_value=self.correct_l2, x_label="Total Batches", y_label="Accuracy (%)", title=r"Accuracy for $L(\Lambda-\alpha\Delta$)", save_plot=True, filename=f"circuit_{self.circuit_type}_accuracy_L2_{self.image_size}x{self.image_size}.png", ylim=[0,1], marker='.')

        epoch_correct += int((correct + correct_2)/2)
        num_batch += 1
        del batch
        del delta
        del g
        del weights_neg
        del weights_pos
        gc.collect()
        
      
      if self.enable_log:
        print(f"The accuracy at epoch {epoch} is {epoch_correct}/ {self.dataset.get_dataset_size()} = {epoch_correct / self.dataset.get_dataset_size()}")
      
      self.accuracy.append(epoch_correct / self.dataset.get_dataset_size())
      
      if output_results:
        tl.generate_plot(y_value=self.accuracy, x_label="Epochs", y_label="Accuracy (%)", title="Model accuracy", save_plot=True, ylim=[0,1], filename=f"circuit_{self.circuit_type}_accuracy_{self.image_size}x{self.image_size}.png", show_max=True, marker='o')
    
    del spsa_v

    gc.collect()


if __name__ == '__main__':
  image_size = 4
  classes = {0: "1", 1: "0"}
  model = Network(image_size=image_size, 
                  circuit_dim=image_size*image_size, 
                  classes=classes, enable_log=True, 
                  draw_circuits=False, epochs=50, 
                  efficient=True, batch=20, 
                  shuffle=True, samples=2000, 
                  shots=1025,
                  circuit_type='experimental',
                  param_a=0.05,
                  param_A=2,
                  param_lambda=0,
                  param_gamma=1,
                  param_eta=1,
                  param_s=0.602,
                  param_t=0.101,
                  param_b=0.01,
                  perf_metrics=True
                  )
  model.train(output_results=True)
  model.save_model()
  model.predict()

  qc = model.get_quantum_operator()
  qc.plot_circuits(4, ["normal", "efficient", "experimental"])
  qc.plot_circuits(8, ["normal", "efficient", "experimental"])
