import os
import cv2
import numpy as np
from os import listdir
from pathlib import Path
from os.path import isfile, join

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt

count_n = 1000
IMG_WIDTH = 128
IMG_HEIGHT = 128

#recreate an image with pixels removed
def create(pp, img):
    
    mask = np.random.choice([0, 1], size=(IMG_WIDTH, IMG_HEIGHT), p=[1-pp, pp])
    idx_w, idx_h = np.where(mask ==  1)
    
    for i in range(len(idx_w)):
        img[idx_w[i],idx_h[i]] = 0
    return img

#extract the images from the dataset
def curate():
    data = Path('/home/wilfred/Datasets/BSR_bsds500/BSR/BSDS500/data/images')
    lst = [x for x in data.iterdir() if data.is_dir()]
    cnt = 0
    for idx, j in enumerate(lst):
        onlyfiles   = [f for f in listdir(lst[idx]) if isfile(join(lst[idx], f))]
        onlyfiles.remove('Thumbs.db')

        for _, i in enumerate(onlyfiles):
            p = join(str(lst[idx]),i)
            img = cv2.imread(p, 0)
            x_patches = image.extract_patches_2d(img, (128,128), max_patches = 2)
            img = create(0.1, img)
            y_patches = image.extract_patches_2d(img, (128,128), max_patches = 2)
            #imgplot = plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
            #plt.show()
            for k in range(len(x_patches)):
                X_train[cnt] = x_patches[k]
                Y_train[cnt] = y_patches[k]
                cnt += 1

#changes need to be made
def printResult():
    y_vari = np.argmax(y_var, axis=3)
    y_pred = model.predict(X_var)
    y_predi = np.argmax(y_pred, axis=3)
    print(y_vari.shape,y_predi.shape)

    for i in range(10,20):
        img = X_var[i]
        segpred = y_predi[i]
        seg = y_vari[i]

        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(img)
        ax.set_title("original")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(design_colormap(segpred,n_classes))
        ax.set_title("FCN")

        ax = fig.add_subplot(1,3,3)
        ax.imshow(design_colormap(seg,n_classes))
        ax.set_title("Ground True")
        plt.show()

    return None 

if __name__ == "__main__":

    #create dataset for images
    ########################## CREATE DATASET #############################
    X_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    Y_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) 
    #######################################################################
    curate()

