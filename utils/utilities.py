import os
import cv2
import numpy as np
from os import listdir
from pathlib import Path
from os.path import isfile, join

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt

#recreate an image with pixels removed
def create(width, height, pp, img):
    
    mask = np.random.choice([0, 1], size=(width, height), p=[1-pp, pp])
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
        cnt += len(onlyfiles)
        for _, i in enumerate(onlyfiles):
            p = join(str(lst[idx]),i)
            img = cv2.imread(p, 0)
            img = create(img.shape[0], img.shape[1], 0.1, img)
            imgplot = plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
            plt.show()

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
    #img = np.random.randint(1, 2, (20,20))
    #print(img)
    #print(create(20, 20, 0.5, img))

    curate()
