import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision = 4)

##1. Start with a data source (Construct a Dataset from the data in the memory)
##2. Create the Dataset
##3. Set batch processing and shuffle
##4. Preprocessing of the dataset elements

def convert(np_array):
  tf_tensor = tf.convert_to_tensor(np_array, dtype=tf.float32)
  return tf_tensor

tf_value = convert(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
dataset = tf.data.Dataset.from_tensor_slices(tf_value)

'''
print(dataset)
############## ITER 1 ###############
for elem in dataset:
    print(elem.numpy())

############## ITER 2 ###############
it = iter(dataset)

print(next(it).numpy().shape)
print(next(it).numpy().shape)


############## DATASET STRUCTURE ###############

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10, 10, 1]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset1.element_spec)
for elem in dataset1:
    print(elem.numpy().shape)

for elem,elem2 in dataset2:
    print(elem.numpy().shape,elem2.numpy().shape)
    print()

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

results = model.fit(train_dataset, epochs=10)
'''

train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images / 255.
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset)

def count(stop):
    i = 0
    while i <  stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
