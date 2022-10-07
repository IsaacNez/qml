import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from utils.dataloader import Dataset
from network.quantum import QuantumOperator
import utils.utils as tl
from joblib import Parallel, delayed

tf.config.set_visible_devices([], 'GPU')

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
                      efficient: bool = False):
  
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
    self.weights            = tf.random.normal((self.circuit_dim - 1, self.unitary_dim ** 2)) if not efficient else tf.random.normal((self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2))
    self.classes            = classes
    self.efficient          = efficient

    self.dataset = Dataset( image_size=self.image_size,
                            enable_transformations=self.enable_transform,
                            enable_log=self.enable_log, filter=True, filter_by=self.classes, samples=-1)

    self.qcircuit = QuantumOperator(circuit_dimension=self.circuit_dim, draw_circuit=draw_circuits, show_gpu_support=False, enable_gpu=False)
  
  def loss(self, prediction, label, classes):
    p_result = max(prediction.values()) / self.shots
    p_label = prediction.pop(classes[label], None) / self.shots
    p_max_false = max(prediction.values()) / self.shots
    return max(p_max_false - p_label + self.param_lambda, 0) ** self.param_eta, 1 if p_result == p_label else 0

  def execute(self, image, label, weights, efficient, classes):
    result = self.qcircuit.execute(image, weights=weights, efficient=efficient)
    loss, correct = self.loss(result, label, classes)
    return loss, correct

  def spsa_loss(self, batch, weights, classes, verbose: bool = False):
    idx_data, idx_label = batch
    total_loss, total_correct, iteration = 0.0, 0, 0

    results = Parallel(n_jobs=12)(delayed(self.execute)(image, label, weights, self.efficient, classes) for image, label in zip(idx_data, idx_label))

    for values in results:
      total_loss += values[0]
      total_correct += values[1]

    if verbose:
      sys.stdout.write("\033[F")
      sys.stdout.write("\033[K")
      print(f"The loss is {total_loss / batch[0].shape[0]} at [{iteration}] with accuracy {total_correct / batch[0].shape[0]} ")
    # for image, label in zip(idx_data, idx_label):
    #   result = self.qcircuit.execute(image, weights=weights, efficient=self.efficient)
    #   loss, correct = self.loss(result, label, classes)
    #   total_loss += loss
    #   total_correct += correct

    #   if verbose:
    #     iteration += 1
    #     sys.stdout.write("\033[F")
    #     sys.stdout.write("\033[K")
    #     print(f"The loss is {loss} at [{iteration}]")
    return (total_loss / batch[0].shape[0]), total_correct

  def train(self, output_results: bool = False):

    spsa_v = tf.zeros((self.circuit_dim - 1, self.unitary_dim ** 2) if not self.efficient else (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2))
    
    self.accuracy = []
    self.loss_g = []

    for epoch in range(self.epochs):
      alpha_k = self.param_a / (epoch + self.param_A + 1) ** self.param_s
      beta_k  = self.param_b / (epoch + 1) ** self.param_t

      epoch_correct = 0

      if self.enable_log:
        print(f"Epoch {epoch}/{self.epochs}")

      num_batch = 0
      for batch in self.dataset.get_batch(batch=self.batch):
        print(f"Batch: {num_batch}\n")
        delta = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.float32).sample(sample_shape=(self.circuit_dim - 1, self.unitary_dim ** 2) if not self.efficient else (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2))
        # delta = tf.random.normal((self.circuit_dim - 1, self.unitary_dim ** 2))
        weights_neg, weights_pos = self.weights - beta_k * delta, self.weights + beta_k * delta

        l_tilde_1, correct = self.spsa_loss(batch, weights_pos, self.classes, True)
        l_tilde_2, _ = self.spsa_loss(batch, weights_neg, self.classes)

        g = (l_tilde_1 - l_tilde_2) / (2 * beta_k)

        self.loss_g.append(g)

        spsa_v = self.param_gamma * spsa_v - g * alpha_k * delta

        self.weights = self.weights + spsa_v

        epoch_correct += correct

        num_batch += 1
        
      
      if self.enable_log:
        print(f"The accuracy at epoch {epoch} is {epoch_correct}/ {self.dataset.get_dataset_size()} = {epoch_correct / self.dataset.get_dataset_size()}")
      self.accuracy.append(epoch_correct / self.dataset.get_dataset_size())
  
    if output_results:
      tl.generate_plot(y_value=self.accuracy, x_label="Epochs", y_label="Accuracy (%)", title="Model accuracy", save_plot=True, filename="accuracy.png")
      tl.generate_plot(y_value=self.loss_g, x_label="Batches", y_label="Modified SPSA Loss", title="Training Convergence", save_plot=True, filename="loss.png")


if __name__ == '__main__':
  image_size = 8
  classes = {0: "0", 1: "1"}
  model = Network(image_size=image_size, circuit_dim=image_size*image_size, classes=classes, enable_log=True, draw_circuits=False, epochs=5, efficient=True, batch=24)
  model.train(output_results=True)


# d = Dataset(image_size=8, enable_transformations=True, enable_log=True, filter=True, filter_by={3: "1", 7: "0"})

# plt.imshow(d.get_image(), cmap='gray')
# plt.show()

# # m = QuantumOperator(circuit_dimension=256, debug_log=True)

# # r = m.execute(d.get_image(), draw=True)

# # print(r)