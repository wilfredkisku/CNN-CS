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
dataset = tf.data.Dataset.from_tensor_slices(tf_value)
print(dataset)
'''
print(dataset)
############## ITER 1 ###############
for elem in dataset:
    print(elem.numpy())

############## ITER 2 ###############
it = iter(dataset)

print(next(it).numpy().shape)
print(next(it).numpy().shape)
'''
############## DATASET STRUCTURE ###############

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10, 10, 1]))

print(dataset1.element_spec)
for elem in dataset1:
    print(elem.numpy().shape)
