import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision = 4)

##1. Start with a data source (Construct a Dataset from the data in the memory)
##2. 

def convert(np_array):
  tf_tensor = tf.convert_to_tensor(np_array, dtype=tf.float32)
  return tf_tensor

tf_value = convert(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

print(tf_value)

dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0,2,(10,100,100,1)))

for elem in dataset:
    print(elem.numpy().shape)
