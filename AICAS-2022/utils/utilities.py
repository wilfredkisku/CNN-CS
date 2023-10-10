import os
import cv2
import copy
import numpy as np
from os import listdir
from pathlib import Path
from os.path import isfile, join
from tqdm import tqdm
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt

probs = [0.5, 0.6, 0.7, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]

#recreate an image with pixels removed
def create(pp, img):
    
    img_ = np.zeros((8, 128, 128, 1), dtype=np.float32)

    for i in range(len(img)):
        mask = np.random.choice([0, 1], size=(128, 128), p=[1-pp, pp])
        idx_w, idx_h = np.where(mask ==  1)
    
        for j in range(len(idx_w)):
            img[i, idx_w[j],idx_h[j],0] = 0
        
        img_[i,:,:,0] = img[i,:,:,0]
    return img_

#extract the images from the dataset
def curate(X_train, Y_train):
    data = Path('/home/wilfred/Datasets/BSR_bsds500/BSR/BSDS500/data/images')
    lst = [x for x in data.iterdir() if data.is_dir()]
    cnt = 0
    for idx, j in tqdm(enumerate(lst)):
        onlyfiles   = [f for f in listdir(lst[idx]) if isfile(join(lst[idx], f))]
        onlyfiles.remove('Thumbs.db')

        for _, i in enumerate(onlyfiles):
            p = join(str(lst[idx]),i)
            img = cv2.imread(p, 0)
            y_patches = image.extract_patches_2d(img, (128,128), max_patches = 8)
            
            y_patches = np.reshape(y_patches,(8, 128,128,-1))

            
            y_patch = copy.deepcopy(y_patches)
            x_patches = create(0.1, y_patch)
            
            for k in range(len(x_patches)):
                X_train[cnt] = x_patches[k]
                Y_train[cnt] = y_patches[k]
                cnt += 1
            
    return X_train, Y_train

def tensorflowDataset(X_train, Y_train):
        dataset = tf.data.Dataset.from_tensor_slice((X_train, Y_train))


def printResult(X, Y):


    fig = plt.figure(figsize=(9, 4))
    columns = 10
    rows = 1
    for i in range(1, columns*rows + 1):
        img_x = X[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img_x, cmap = 'gray')
    plt.show()
    return None 

def saveResult():
    rows = 3
    cols = 3

    fig =  plt.figure(figsize=(12,10))

    for i in range(1,rows+1):
        for j in range(1, cols+1):
            img = cv2.imread(os.path.join(os.getcwd(),onlyfewfiles[((i-1)*rows)+j - 1]), 0)
            patch = image.extract_patches_2d(img, (128,128), max_patches = 1)
            img = np.resize(patch,(128,128))
            ax = fig.add_subplot(rows, cols, ((i-1)*rows)+j)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{}\nNewline'.format(onlyfewfiles[((i-1)*rows)+j - 1]), fontsize=8)
            plt.imshow(img, cmap='gray')

    plt.show()

    return None

if __name__ == "__main__":

    count_n = 500 * 8
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    #create dataset for images
    ########################## CREATE DATASET #############################
    X_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8) 
    #######################################################################
    X, Y = curate(X_train, Y_train)

    #######################################################################
    number_of_rows = X.shape[0]
    random_indices = np.random.choice(number_of_rows, size=10, replace=False)
    X_random_rows = X[random_indices, :, :, :]
    Y_random_rows = Y[random_indices, :, :, :]
    #######################################################################

    print(X_random_rows.shape)
    print(Y_random_rows.shape)

    printResult(X_random_rows,Y_random_rows)

    plt.imshow(X_random_rows[0], cmap='gray')
    plt.show()

    plt.imshow(Y_random_rows[0], cmap='gray')
    plt.show()

    '''
    plt.imshow(Y[0], cmap='gray')
    plt.show()

    plt.imshow(X[0], cmap='gray')
    plt.show()
    '''
