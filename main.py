import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import sklearn.model_selection import train_test_split

def network(x_train, y_train, x_val, y_val):

    input_img = tf.keras.Input(shape=(128, 128, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    ae = tf.keras.Model(input_img, decoded)
    ae.compile(optimizer='adam', loss='MSE')

    ae.summary()

    history = ae.fit(x_train, y_train, epochs=35, batch_size=64, shuffle=True, validation_data=(x_test, y_test), verbose = 2)


if __name__ == "__main__":
    #Get the curated data from the dataset
    X, Y = utilities.curate()
    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, Y, test_size = 0.2)
    network(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
