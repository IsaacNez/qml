from genericpath import isdir
import numpy as np
from typing import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

# Enable the Agg backend to avoid 
# figure artifacts being left behind
# after saving an image in non-interactive mode
# This will reduce memory consumption by main process
# and subsequentely, it will avoid crashing due to running
# out of RAM.
# Source: https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close
matplotlib.use('Agg')

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
                  fontsize: int = 36,
                  save_plot: bool = False,
                  filename: str = None,
                  xlim: list = None,
                  ylim: list = None,
                  path: str = 'output',
                  **kwargs) -> None:

  if x_value is None and y_value is None:
    sys.exit("Please provide at least the y_value")
  
  plt.figure(figsize=[20, 15])
  if x_value is not None:
    plt.plot(x_value, y_value, **kwargs)
  else:
    plt.plot(y_value, **kwargs)
  
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.title("Generic Plot Title" if title is None else title, fontsize=fontsize)
  plt.xlabel("X-axis values" if x_label is None else x_label, fontsize=fontsize)
  plt.ylabel("Y-axis values" if y_label is None else y_label, fontsize=fontsize)
  
  if xlim is not None:
    plt.xlim(xlim)

  if ylim is not None:
    plt.ylim(ylim)

  if not save_plot and filename is None:
    plt.show(block=False)
    plt.cla()
    plt.clf()
    plt.close('all')
  else:
    if not os.path.exists(path):
      os.makedirs(path)
    output_path = os.path.abspath(path)
    plt.savefig(os.path.join(output_path, filename), format="png")
    plt.cla()
    plt.clf()
    plt.close('all')

  
