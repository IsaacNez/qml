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
  """ Generates a random integer

  Args:
    len:      Upper limit for the random generator
    start:    Lower limit for the random generator
  
  Output:
    int:      Generates a random integer between start and len.
  """
  return np.random.randint(len - start) + start


def generate_random_range(len: int) -> Tuple[int, int]:
  """ Generates a random range 

      Creates the start and end limits of a range
      by using a random generator.

  Args:
    len:                Represents the amount of integers for the range

  Output:
    Tuple[int, int]:    A range with random limits of length len
  """
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
                  show_max: bool = False,
                  **kwargs) -> None:

  """ Plotting interface for the Network class.

  Args:
    x_value:        Values for the x-axis. x_value and y_value should have the same length.
    
    y_value:        Values for the y-axis. x_value and y_value should have the same length.
                    If x_value is not given, it will be assumed from the length of y_value.

    x_label:        The label for the x-axis.

    y_label:        The label for the y-axis.

    title:          A title for the figure.

    fontsize:       The fontsize for the elements in the figure.

    save_plot:      Indicates if the figure should be interactively displayed or saved to file.

    filename:       Filename for the figure when it must be saved to file.

    xlim:           Range limit for the x-axis.

    ylim:           Range limit for the y-axis.

    path:           Path where the figure should be saved.

    show_max:       If the plot should show a line representing where the max value is.

    kwargs:         Another arguments passed down to the Figure class.

  Raises:
    ValueError:     If x_value and y_value are not provided, it will raised a ValueError exception.
  """
  if x_value is None and y_value is None:
    raise ValueError("Please provide at least the y_value")
  
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

  num_points = len(y_value)
  if show_max and num_points > 4:
    max_y = max(y_value)
    half_x = num_points // 2
    plt.axhline(y=max_y, color="red", linestyle="--")
    plt.annotate(f"{max_y}",
      xy=(half_x, max_y),
      xytext=(half_x+0.5*half_x, max_y + 0.5*max_y),
      color="red",
      fontsize=fontsize,
      arrowprops=dict(facecolor="red")
    )


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

  
