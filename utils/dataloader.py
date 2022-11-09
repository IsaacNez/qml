"""Definition of the MNIST Dataset Handler Class"""
from typing import *
import numpy as np
import tensorflow as tf

from utils.utils import *

tf.config.set_visible_devices([], 'GPU')

class Dataset():
  """MNIST Dataset Class"""

  def __init__(self,
              image_size: int = 4,
              enable_transformations: bool = False,
              batch: int = 222, 
              enable_log: bool = False,
              filter: bool = False,
              filter_by: dict = None,
              samples: int = 100,
              shuffle: bool = False) -> 'Dataset':
    """ Download the MNIST Dataset to be used for
        training, test, and validation.
    
    Args:
      image_size:               Defines the size of the image to be scaled to.
                                Default is 4, thus the original images will be
                                scaled from 28x28 to 4x4. Values greater than 28
                                are not accepted.

      enable_transformations:   Enables transformations for the MNIST dataset to 
                                improve training. Default is set to False.

      batch:                    Defines the batch size. Default is 222

      enable_log:               Enables verbose loggin for this class

      filter:                   Filter the dataset to only have the classes provided
                                by filter_by parameter. 
      
      filter_by:                Dictionary with the classes to filter by.
                                The dictionary should have the format:
                                {<digit>:"<class>", <digit>:"<class>"}
                                Digit is an integer from 0 to 9 and class should be
                                0 or 1.
                              
      samples:                  Specify how many samples you want to use to train and
                                test. From samples, 0.8*samples will be used to train 
                                and 0.2*samples will be used test from their respective
                                dataset. When samples is -1, it will use the complete train
                                and test dataset.

      shuffle:                  Indicate if you wish to shuffle the dataset before calling
                                get_batch. This to ensure each epoch obtains randomly sorted
                                images.
    """
    
    if not isinstance(image_size, int):
      raise TypeError("Please provide an integer for the image size")
    elif image_size % 2 != 0:
      raise ValueError("Please provide an even integer for the image size")
    elif image_size > 28 or image_size < 4:
      raise ValueError("Please an image size between 4 and 28")

    if filter and filter_by is None:
      raise ValueError("If you want to filter the dataset, you need to specify the classes to filter by")
    elif filter and len(filter_by.keys()) > 2:
      raise ValueError("This dataset only supports two classes.")

    
    self.image_size = image_size
    self.batch = batch
    self.shuffle = shuffle

    (self.image_train, self.label_train), (self.image_test, self.label_test) = tf.keras.datasets.mnist.load_data()


    even_samples = samples - samples % 2
    half_idx = even_samples // 2
    train_idx = int(0.8*half_idx) if samples > 0 else None
    test_idx = int(0.2*half_idx) if samples > 0 else None


    if filter:
      keys = list(filter_by.keys())

      idxs_train = np.append(tf.where(tf.equal(self.label_train, keys[0])).numpy()[:train_idx], 
                              tf.where(tf.equal(self.label_train, keys[1])).numpy()[:train_idx])
      idxs_test = np.append(tf.where(tf.equal(self.label_test, keys[0])).numpy()[:test_idx], 
                                    tf.where(tf.equal(self.label_test, keys[1])).numpy()[:test_idx])

      self.image_train, self.label_train = self.image_train[idxs_train], self.label_train[idxs_train]
      self.image_test, self.label_test = self.image_test[idxs_test], self.label_test[idxs_test]

    self.image_train, self.image_test = self.image_train[..., np.newaxis] / 255.0 , self.image_test[..., np.newaxis] / 255.0

    image_train, image_test = tf.image.resize(self.image_train, (self.image_size, self.image_size)), tf.image.resize(self.image_test, (self.image_size, self.image_size))

    if enable_transformations:
      cropped_train, rotated_train, cropped_test, rotated_test = self.transforms(self.image_train, self.image_test)

      self.image_train = image_train
      self.image_train = tf.concat([self.image_train, cropped_train], axis=0)
      self.image_train = tf.concat([self.image_train, rotated_train], axis=0)

      self.image_test = image_test
      self.image_test = tf.concat([self.image_test, cropped_test], axis=0)
      self.image_test = tf.concat([self.image_test, rotated_test], axis=0)
    else:
      self.image_train, self.image_test = image_train, image_test

    # Define the idxs here to avoid the same random generation when shuffling them for mini-batch
    self.indices = tf.range(start=0, limit=self.image_train.shape[0], dtype=tf.int32)
    
    if enable_log:
      print(f"The resulting training dataset has {self.image_train.shape[0]} images and the test one has {self.image_test.shape[0]} images")


  def transforms(self, image_train: tf.Tensor, image_test: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """ Applies random crop and 90 degrees rotation to a subset of the MNIST Dataset

    Args:
      image_train:  A 3D Tensor array for the train dataset

      image_test:   A 3D Tensor for the test dataset
    
    Output:
      cropped:        A numpy ndarray with the cropped images of size mxm. 
                      The size matches the resize size of this class.

      rotated:        The rotated (and resized) images.

      crooped_test:   The same as cropped but for the test dataset.

      rotated_test:   The same as rotated but for the test dataset.
    """
    start, end = generate_random_range(image_train.shape[0])
    cropped = tf.image.random_crop(image_train[start:end, :, :, :], size=(end-start, self.image_size, self.image_size, 1))
    self.label_train = np.append(self.label_train, self.label_train[start:end])
    cropped = tf.cast(cropped, tf.float32)

    start, end = generate_random_range(image_test.shape[0])
    cropped_test = tf.image.random_crop(image_test[start:end, :, :, :], size=(end-start, self.image_size, self.image_size, 1))
    self.label_test = np.append(self.label_test, self.label_test[start:end])
    cropped_test = tf.cast(cropped_test, tf.float32)


    start, end = generate_random_range(image_train.shape[0])
    rotated = tf.image.rot90(image_train[start:end, :, :, :])
    self.label_train = np.append(self.label_train, self.label_train[start:end])

    start, end = generate_random_range(image_test.shape[0])
    rotated_test = tf.image.rot90(image_test[start:end, :, :, :])
    self.label_test = np.append(self.label_test, self.label_test[start:end])

    rotated, rotated_test =  tf.image.resize(rotated, (self.image_size, self.image_size)), \
                                                  tf.image.resize(rotated_test, (self.image_size, self.image_size)),

    return cropped, rotated, cropped_test, rotated_test

  def get_batch(self, batch: int = 222) -> Iterable[Tuple[tf.Tensor, np.ndarray]]:
    """ Returns an interable batch of images for training
    
    Args:
      batch:                        Indicates the size of the batch

    Output:
      Iterable[Tensor, NDArray]:    An iterable batch for images and the corresponding
                                    labels. The size of Tensor is (batch, image_size, image_size)
                                    The size of NDArray is (batch,)   
    """
    self.shuffle_train_dataset()

    for idx in range(0, self.image_train.shape[0], batch):
      lim = min(idx + batch, self.image_train.shape[0])
      idxs = slice(idx, lim)
      yield tf.convert_to_tensor(self.image_train[idxs]), self.label_train[idxs]

  def shuffle_train_dataset(self) -> None:
    """ Shuffle train dataset and labels accordingly.
    """
    if self.shuffle:
      self.indices = tf.random.shuffle(self.indices)

      self.image_train = tf.gather(self.image_train, self.indices).numpy() 
      self.label_train = tf.gather(self.label_train, self.indices).numpy()

  def get_image(self, index: int = 0) -> tf.Tensor:
    """ Returns an specific image from the training dataset.

    Args:
      index:        Index of image to obtain from the training dataset.
    
    Output:
      tf.Tensor     Tensor representing the image at position index 
                    from the training dataset. The size of the Tensor
                    is (1, image_size, image_size)
    """
    if index > self.image_train.shape[0]:
      raise ValueError("Index out of range for the training dataset")
    return self.image_train[index]

  def get_dataset_size(self) -> int:
    """ Obtain the training dataset from the Tensor storing it. """
    return self.image_train.shape[0]

  def get_test_samples(self) -> Iterable[Tuple[tf.Tensor, str]]:
    """ Iterator for the testing images.

    Ouput:
      Iterable[tf.Tensor, str]:   Yields the image and the corresponding 
                                  label. The size of the Tensor is 
                                  (1, image_size, image_size)
    """
    for image, label in zip(self.image_test, self.label_test):
      yield image, label
  
  def get_test_dataset_size(self) -> int:
    """ Returns the test dataset size """
    return self.image_test.shape[0]
