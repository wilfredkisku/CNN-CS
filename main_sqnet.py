import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

def curateData():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def psnr_mean(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_loss(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))) + tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))

def fire(x, squeeze, expand):
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
    return tf.keras.layers.concatenate([y1, y3])

def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)

def SimpleCSNet_sq():
    input_layer = tf.keras.layers.Input(shape=[128, 128, 1])
    y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)
    y = fire_module(32, 32)(y)
    y = fire_module(32, 32)(y)
    y = fire_module(32, 32)(y)
    y = fire_module(32, 32)(y)
    y = fire_module(32, 32)(y)
    output_layer = tf.keras.layers.Conv2D(1, (1,1), padding = 'same', activation='sigmoid')(y)

    ae = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    ae.compile(optimizer='adam', loss=ssim_loss, metrics = [psnr_mean])
    ae.summary()

    return ae

if __name__ == "__main__":
    
    #########################################################
    ################ TRAINING AND VALIDATING ################
    #########################################################

    count_n = 500 * 10
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    X_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

    X, Y = curateData(X_train, Y_train)

    X = X / 255.
    Y = Y / 255.

    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, Y, test_size = 0.2, random_state = 42)

    model_cnn = SimpleCSNet_sq()

    checkpointer = tf.keras.callbacks.ModelCheckpoint('/workspace/data/cs-sq-1000.h5', verbose = 1, save_best_only = True)
    history = ae.fit(X_TRAIN, Y_TRAIN, epochs = 1000, batch_size = 32, shuffle=True, validation_data = (X_VAL, Y_VAL), verbose =1 , callbacks=[checkpointer])

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('/workspace/data/cs-sq-history-1000.csv')
    print('End of training ...')
