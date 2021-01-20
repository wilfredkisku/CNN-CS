import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utilities import curate

count_n = 500 * 4
IMG_WIDTH = 128
IMG_HEIGHT = 128

def network(x_train, y_train, x_val, y_val):

    input_img = tf.keras.layers.Input(shape=(128, 128, 1))
    
    #input_norm = tf.keras.layers.Lambda(lambda x:x / 255)(input_img)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    ae = tf.keras.models.Model(inputs = [input_img], outputs = [decoded])
    ae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    ae.summary()

    history = ae.fit(x_train, y_train, epochs=35, batch_size=64, shuffle=True, validation_data=(x_val, y_val), verbose = 2)


if __name__ == "__main__":
    #create dataset for images
    ########################## CREATE DATASET #############################
    X_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
    #######################################################################
    #Get the curated data from the dataset
    
    X, Y = curate(X_train, Y_train)
    
    '''
    plt.imshow(Y[0], cmap='gray')
    plt.show()

    plt.imshow(X[0], cmap='gray')
    plt.show()
    '''
    
    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, Y, test_size = 0.2, random_state = 42)
   
    '''
    print(X_TRAIN.shape)
    print(X_VAL.shape)
    print(Y_TRAIN.shape)
    print(Y_VAL.shape)

    plt.imshow(X_TRAIN[0])
    plt.show()
    plt.imshow(Y_TRAIN[0])
    plt.show()

    plt.imshow(X_VAL[0])
    plt.show()
    plt.imshow(Y_VAL[0])
    plt.show()
    '''
    network(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    
