import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utilities_new import curate

def psnr_mean(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_loss(true, pred):
    return (1 - tf.reduce_mean(tf.image.ssim(true, pred, 1.0)))

def SimpleCSNet(x_train, y_train, x_val, y_val):

    input_layer = tf.keras.layers.Input((128, 128, 1))
    
    conv1 = tf.keras.layers.Conv2D(8, (3,3), padding='same')(input_layer)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(8, (3,3), padding='same')(conv1)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D(16, (3,3), padding='same')(conv2)
    conv3 = tf.keras.layers.ReLU()(conv3)

    conv4 = tf.keras.layers.Conv2D(16, (3,3), padding='same')(conv3)
    conv4 = tf.keras.layers.ReLU()(conv4)

    conv5 = tf.keras.layers.Conv2D(32, (3,3), padding='same')(conv4)
    conv5 = tf.keras.layers.ReLU()(conv5)

    conv6 = tf.keras.layers.Conv2D(32, (3,3), padding='same')(conv5)
    conv6 = tf.keras.layers.ReLU()(conv6)

    output_layer = tf.keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(conv6)

    ae = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    ae.compile(optimizer='adam', loss=ssim_loss)

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
        
    X, Y = curate(X_train, Y_train)
    
    X = X / 255.
    Y = Y / 255.

    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X, Y, test_size = 0.2, random_state = 42)
   
    model_cnn = SimpleCSNet()
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint('/workspace/data/cs-hi-1000.h5', verbose=1, save_best_only=True)
    history = model_cnn.fit(X_TRAIN, Y_TRAIN, epochs=1000, batch_size=32, shuffle=True, validation_data=(X_VAL, Y_VAL), verbose = 1,  callbacks=[checkpointer])

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('/workspace/data/cs-hi-history-1000.csv')
    print('End of training ...')

    predict = model_cnn.predict(X_VAL[:10,:,:,:])
    x_val = X_VAL[:10,:,:,:]
    y_val = Y_VAL[:10,:,:,:]

    fig = plt.figure(figsize=(9, 3))
    columns = 10
    rows = 1
    for i in range(1, columns*rows + 1):
        img_x = predict[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img_x, cmap = 'gray')
    plt.show()

