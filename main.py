import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utilities_new import curate

count_n = 500 * 10
IMG_WIDTH = 128
IMG_HEIGHT = 128

def network(x_train, y_train, x_val, y_val):

    input_img = tf.keras.layers.Input(shape=(128, 128, 1))
    
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    y = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    
    ae = tf.keras.models.Model(inputs = [input_img], outputs = [decoded])
    ae.compile(optimizer='adam', loss=ssim_loss)

    ae.summary()

    history = ae.fit(x_train, y_train, epochs=1000, batch_size=128, shuffle=True, validation_data=(x_val, y_val), verbose = 1)


if __name__ == "__main__":
    
    #####################################################################
    ########################## CREATE DATASET ###########################
    #####################################################################

    X_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
        
    X, Y = curate(X_train, Y_train)
    
    X = X / 255
    Y = Y / 255

    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    network(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    
