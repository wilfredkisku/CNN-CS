import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utilities import curate

count_n = 500 * 8
IMG_WIDTH = 128
IMG_HEIGHT = 128

def networkNew(x_train, y_train, x_val, y_val):

    input_layer = tf.keras.layers.Input((128, 128, 1))
    conv1 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(input_layer)
    conv1 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

    conv3 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)

    conv4 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2,2))(conv4)

    #Middle
    convm = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(pool4)
    convm = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(convm)

    #upconv part
    deconv4 = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv4)

    deconv3 = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv3)

    deconv2 = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    uconv2 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv2)

    deconv1 = tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(uconv1)

    output_layer = tf.keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(uconv1)

    ae = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    ae.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])

    ae.summary()

    history = ae.fit(x_train, y_train, epochs=35, batch_size=64, shuffle=True, validation_data=(x_val, y_val), verbose = 1)

def network(x_train, y_train, x_val, y_val):

    input_img = tf.keras.layers.Input(shape=(128, 128, 1))
    
    #input_norm = tf.keras.layers.Lambda(lambda x:x / 255)(input_img)

    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    #x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    #x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    #x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    #x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    ae = tf.keras.models.Model(inputs = [input_img], outputs = [decoded])
    ae.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])

    ae.summary()

    history = ae.fit(x_train, y_train, epochs=35, batch_size=64, shuffle=True, validation_data=(x_val, y_val), verbose = 1)


if __name__ == "__main__":
    #create dataset for images
    ########################## CREATE DATASET #############################
    X_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, 128, 128, 1), dtype=np.uint8)
    #######################################################################
    #Get the curated data from the dataset
        
    X, Y = curate(X_train, Y_train)
    
    X = X / 255
    Y = Y / 255

    #print((X[0]))

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
    #network(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    networkNew(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
