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

#recreate an image with pixels removed
def create(pp, img):
    
    img_ = np.zeros((4, 128, 128, 1), dtype=np.float32)

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
            y_patches = image.extract_patches_2d(img, (128,128), max_patches = 4)
            
            y_patches = np.reshape(y_patches,(4, 128,128,-1))

            
            y_patch = copy.deepcopy(y_patches)
            x_patches = create(0.1, y_patch)
            
            for k in range(len(x_patches)):
                X_train[cnt] = x_patches[k]
                Y_train[cnt] = y_patches[k]
                cnt += 1
            
    return X_train, Y_train


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

    count_n = 500 * 4
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    #create dataset for images
    ########################## CREATE DATASET #############################
    X_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y_train = np.zeros((count_n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8) 
    #######################################################################
    X, Y = curate(X_train, Y_train)

    print(Y[0].shape)
    print(X[0].shape)

    plt.imshow(Y[0], cmap='gray')
    plt.show()

    plt.imshow(X[0], cmap='gray')
    plt.show()

