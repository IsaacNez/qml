import numpy as np
from typing import *
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

def generate_random_int(len: int, start: int = 0) -> int:
  return np.random.randint(len - start) + start


def generate_random_range(len: int) -> Tuple[int, int]:
  start = generate_random_int(len)
  end = generate_random_int(len, start)
  return start, end

def generate_plot(x_value: list or np.ndarray or tf.Tensor = None,
                  y_value: list or np.ndarray or tf.Tensor = None,
                  x_label: str = None,
                  y_label: str = None,
                  title: str = None,
                  save_plot: bool = False,
                  filename: str = None,
                  **kwargs) -> None:

  if x_value is None and y_value is None:
    sys.exit("Please provide at least the y_value")
  
  plt.figure(figsize=[19.2, 14.4])
  if x_value is not None:
    plt.plot(x_value, y_value, **kwargs)
  else:
    plt.plot(y_value, **kwargs)
  
  plt.title("Generic Plot Title" if title is None else title)
  plt.xlabel("X-axis values" if x_label is None else x_label)
  plt.ylabel("Y-axis values" if y_label is None else y_label)
  
  if not save_plot and filename is None:
    plt.show(block=False)
    plt.close()
  else:
    output_path = os.path.abspath("output/")
    plt.savefig(os.path.join(output_path, filename), format="png")
    plt.close()

  
