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

def SimpleCSNet2(x_train, y_train, x_val, y_val):

    input_layer = tf.keras.layers.Input((128, 128, 1))
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    ae = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    ae.compile(optimizer='adam', loss=ssim_loss)

    ae.summary()

    return ae

if __name__ == "__main__":
    
    #########################################################
    ################ TRAINING AND VALIDATING ################
    #########################################################
    
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

    count_n = 0

    for i in range(len(lst_)):
        count_n += len(os.listdir(os.path.join(Path2,lst_[i])))

    X_train = np.zeros((count_n * 2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n * 2, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

    curate_(Path2, lst_, X_train, Y_train)

    X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(np.concatenate((x_train,X_train),axis=0), np.concatenate((y_train,Y_train),axis=0), test_size = 0.2, random_state = 42)
   
    model_cnn = SimpleCSNet2()
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint('/workspace/data/cs-model-1000.h5', verbose=1, save_best_only=True)
    history = model_cnn.fit(X_TRAIN, Y_TRAIN, epochs=1000, batch_size=8, shuffle=True, validation_data=(X_VAL, Y_VAL), verbose = 1,  callbacks=[checkpointer])

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('/workspace/data/cs-history-1000.csv')
    
    print('End of training ...')

'''
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
'''
