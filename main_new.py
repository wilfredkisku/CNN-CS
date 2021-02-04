import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils.utilities import curate, curate_

import cv2
import copy
from os import listdir
from pathlib import Path
from os.path import isfile, join
from tqdm import tqdm
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

def psnr_mean(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def SimpleCSNet():

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
    ae.compile(optimizer='adam', loss=ssim_loss, metrics = [psnr_mean])

    ae.summary()

    return ae

if __name__ == "__main__":
    
    
    count_n = 0
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    Path1 = Path('/workspace/storage/cnn-cs/data/images')
    Path2 = Path('/workspace/storage/cnn-cs/data/train')

    lst  = [x for x in Path1.iterdir() if Path1.is_dir()]
    lst_ = [x for x in Path2.iterdir() if Path2.is_dir()]

    for i in range(len(lst)):
        count_n += len(os.listdir(os.path.join(Path1,lst[i]))) - 1

    x_train = np.zeros((count_n * 8, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    y_train = np.zeros((count_n * 8, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

    curate(Path1, lst, x_train, y_train)

    x_train = x_train / 255.
    y_train = y_train / 255.

    count_n = 0

    for i in range(len(lst_)):
        count_n += len(os.listdir(os.path.join(Path2,lst_[i])))

    X_train = np.zeros((count_n * 2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n * 2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

    curate_(Path2, lst_, X_train, Y_train)

    X_train = X_train / 255.
    Y_train = Y_train / 255.

    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(np.concatenate((x_train,X_train),axis=0), np.concatenate((y_train,Y_train),axis=0), test_size = 0.2, random_state = 42)

    model_cnn = SimpleCSNet()
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint('/workspace/data/cs-hi-model-1000.h5', verbose=1, save_best_only=True)
    history = model_cnn.fit(X_TRAIN, Y_TRAIN, epochs=500, batch_size=64, shuffle=True, validation_data=(X_VAL, Y_VAL), verbose = 1,  callbacks=[checkpointer])

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('/workspace/data/cs-hi-history-1000.csv')
    print('End of training ...')

