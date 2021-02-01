import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

def curateData():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def fire(x, squeeze, expand):
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
    return tf.keras.layers.concatenate([y1, y3])

def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)

x = tf.keras.layers.Input(shape=[128, 128, 1])
y = tf.keras.layers.Conv2D(kernel_size=3, filters=12, padding='same', use_bias=True, activation='relu')(x)
y = fire_module(12, 12)(y)
y = fire_module(12, 12)(y)
y = fire_module(12, 12)(y)
y = fire_module(12, 12)(y)
y = fire_module(12, 12)(y)
y = tf.keras.layers.Conv2D(1, (1,1), padding = 'same', activation='sigmoid')(y)

model = tf.keras.Model(x, y)
model.summary()

X_train, Y_train, X_test, Y_test = curateData()

'''
img = X_train[0]
plt.imshow(img, cmap='gray')
plt.show()
'''
