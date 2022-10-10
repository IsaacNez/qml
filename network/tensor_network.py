import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import operator
import tensorflow as tf
import tensorflow_probability as tfp
from utils.dataloader import Dataset
from network.quantum import QuantumOperator
import utils.utils as tl
from joblib import Parallel, delayed
from itertools import cycle

devices = tf.config.list_physical_devices()
for device in devices:
  if device.device_type == 'GPU':
    tf.config.experimental.set_memory_growth(device, True)

# tf.config.set_visible_devices([], 'GPU')

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
                      shuffle: bool = False,
                      samples: int = -1):
  
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
    self.shuffle            = shuffle

    self.dataset = Dataset( image_size=self.image_size,
                            enable_transformations=self.enable_transform,
                            enable_log=self.enable_log, filter=True, filter_by=self.classes, samples=samples, shuffle=self.shuffle)

    self.qcircuit = QuantumOperator(circuit_dimension=self.circuit_dim, draw_circuit=draw_circuits, show_gpu_support=False, enable_gpu=False)
  
  def loss(self, prediction, label, classes):
    p_result = max(prediction.values()) / self.shots
    p_label = prediction.pop(classes[label], None) / self.shots
    p_max_false = max(prediction.values()) / self.shots
    return max(p_max_false - p_label + self.param_lambda, 0) ** self.param_eta, 1 if p_result == p_label else 0

  def execute(self, image, label, weights, efficient, classes, device):
    result = self.qcircuit.execute(image, weights=weights, efficient=efficient, device=device, shots=self.shots)
    loss, correct = self.loss(result, label, classes)
    return loss, correct

  def spsa_loss(self, batch, weights, classes, verbose: bool = False, name: str = "SPSA Loss"):
    idx_data, idx_label = batch
    total_loss, total_correct = 0.0, 0
    
    jobs = os.cpu_count()
    
    if len(devices) > 1:
      jobs += len(devices) - 1
    results = Parallel(n_jobs=int(jobs))(delayed(self.execute)(image, label, weights, self.efficient, classes, device) for image, label, device in zip(idx_data, idx_label, cycle(devices)))

    for values in results:
      total_loss += values[0]
      total_correct += values[1]

    if verbose:
      print(f"The {name} is {total_loss / batch[0].shape[0]} with accuracy {total_correct / batch[0].shape[0]} ")

    return (total_loss / batch[0].shape[0]), total_correct

  def predict(self):

    if self.enable_log:
      print("Starting testing...")
    jobs = os.cpu_count()
    
    if len(devices) > 1:
      jobs += len(devices) - 1

    results = Parallel(n_jobs=int(jobs))(delayed(self.qcircuit.execute)(image=sample[0], weights=self.weights, efficient=self.efficient, device=device, shots=self.shots) for sample, device in zip(self.dataset.get_test_samples(), cycle(devices)))
    
    prediction_correct = 0

    for prediction, sample in zip(results, self.dataset.get_test_samples()):
      prediction_class = max(prediction.items(), key=operator.itemgetter(1))[0]
      prediction_label = list(self.classes.keys())[list(self.classes.values()).index(prediction_class)]
      correct_label = sample[1]
      print(f"<{prediction}>:Pred. class=[{prediction_class}], Correct Label=[{correct_label}], Pred. Label=[{prediction_label}]", end="")
      if prediction_label == correct_label:
        print(" \u2714")
        prediction_correct += 1
      else:
        print(" \u2716")
    print(f"The accuracy of the model is {prediction_correct}/{self.dataset.get_test_dataset_size()}={prediction_correct/self.dataset.get_test_dataset_size()}")


  def train(self, output_results: bool = False):

    spsa_v = tf.zeros((self.circuit_dim - 1, self.unitary_dim ** 2) if not self.efficient else (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2))
    
    self.accuracy = []
    self.loss_g = []
    self.loss_l1 = []
    self.loss_l2 = []
    self.correct_l1 = []
    self.correct_l2 = []

    for epoch in range(self.epochs):
      alpha_k = self.param_a / (epoch + self.param_A + 1) ** self.param_s
      beta_k  = self.param_b / (epoch + 1) ** self.param_t

      epoch_correct = 0

      if self.enable_log:
        print(f"Epoch {epoch}/{self.epochs}")

      num_batch = 0
      for batch in self.dataset.get_batch(batch=self.batch):
        print(f"Batch: {num_batch}")
        delta = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.float32).sample(sample_shape=(self.circuit_dim - 1, self.unitary_dim ** 2) if not self.efficient else (self.circuit_dim - 3, (self.unitary_dim ** 2) ** 2))
        # delta = tf.random.normal((self.circuit_dim - 1, self.unitary_dim ** 2))
        weights_neg, weights_pos = self.weights - beta_k * delta, self.weights + beta_k * delta

        l_tilde_1, correct = self.spsa_loss(batch, weights_pos, self.classes, True, "L1 Loss")
        l_tilde_2, correct_2 = self.spsa_loss(batch, weights_neg, self.classes, True, "L2 Loss")

        g = (l_tilde_1 - l_tilde_2) / (2 * beta_k)
        
        self.loss_g.append(g)
        self.loss_l1.append(l_tilde_1)
        self.loss_l2.append(l_tilde_2)
        self.correct_l1.append(correct / batch[0].shape[0])
        self.correct_l2.append(correct_2 / batch[0].shape[0])
        # if self.enable_log:
        #   print(f"The loss g is {self.loss_g}, compared to +alpha {self.loss_l1} and -alpha {self.loss_l2} with acc {correct / batch[0].shape[0]} and {correct_2 / batch[0].shape[0]} respectively")
        spsa_v = self.param_gamma * spsa_v - g * alpha_k * delta

        self.weights = self.weights + spsa_v

        
        if output_results:
          tl.generate_plot(y_value=self.loss_g, x_label="Total Batches", y_label="Modified SPSA Loss", title="Training Convergence", save_plot=True, filename=f"loss_{self.image_size}x{self.image_size}.png", marker='o')
          tl.generate_plot(y_value=self.loss_l1, x_label="Total Batches", y_label="SPSA Loss +alpha", title="Total Loss", save_plot=True, filename=f"total_loss_L1_{self.image_size}x{self.image_size}.png", marker='o')
          tl.generate_plot(y_value=self.loss_l2, x_label="Total Batches", y_label="SPSA Loss -alpha", title="Total Loss", save_plot=True, filename=f"total_loss_L2_{self.image_size}x{self.image_size}.png", marker='o')
          tl.generate_plot(y_value=self.correct_l1, x_label="Total Batches", y_label="Accuracy", title="Accuracy for L +alpha", save_plot=True, filename=f"accuracy_L1_{self.image_size}x{self.image_size}.png", marker='o')
          tl.generate_plot(y_value=self.correct_l2, x_label="Total Batches", y_label="Accuracy", title="Accuracy for L -alpha", save_plot=True, filename=f"accuracy_L2_{self.image_size}x{self.image_size}.png", marker='o')

        epoch_correct += int((correct + correct_2)/2)
        num_batch += 1

        
      
      if self.enable_log:
        print(f"The accuracy at epoch {epoch} is {epoch_correct}/ {self.dataset.get_dataset_size()} = {epoch_correct / self.dataset.get_dataset_size()}")
      
      self.accuracy.append(epoch_correct / self.dataset.get_dataset_size())
      
      if output_results:
        tl.generate_plot(y_value=self.accuracy, x_label="Epochs", y_label="Accuracy (%)", title="Model accuracy", save_plot=True, filename=f"accuracy_{self.image_size}x{self.image_size}.png", marker='o')


if __name__ == '__main__':
  image_size = 8
  classes = {0: "1", 1: "0"}
  model = Network(image_size=image_size, circuit_dim=image_size*image_size, classes=classes, enable_log=True, draw_circuits=False, epochs=2, efficient=True, batch=222, shuffle=True, samples=-1, shots=1024)
  model.train(output_results=True)
  model.predict()


# d = Dataset(image_size=8, enable_transformations=True, enable_log=True, filter=True, filter_by={3: "1", 7: "0"})

# plt.imshow(d.get_image(), cmap='gray')
# plt.show()

# # m = QuantumOperator(circuit_dimension=256, debug_log=True)

# # r = m.execute(d.get_image(), draw=True)

# # print(r)
