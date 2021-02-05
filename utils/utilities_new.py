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

def create(pp, img, p):

    img_ = np.zeros((p, 128, 128, 1), dtype=np.float32)

    for i in range(len(img)):
        mask = np.random.choice([0, 1], size=(128, 128), p=[1-pp, pp])
        idx_w, idx_h = np.where(mask ==  1)
    
        for j in range(len(idx_w)):
            img[i, idx_w[j],idx_h[j],0] = 0
        
        img_[i,:,:,0] = img[i,:,:,0]
    return img_

def curate(path, lt,  x, y):
    
    cnt = 0
    for idx, j in tqdm(enumerate(lt)):
        onlyfiles   = [f for f in listdir(lt[idx]) if isfile(join(lt[idx], f))]
        onlyfiles.remove('Thumbs.db')

        for _, i in enumerate(onlyfiles):
            p = join(str(lt[idx]),i)
            img = cv2.imread(p, 0)
            y_patches = image.extract_patches_2d(img, (128,128), max_patches = 8)
            
            y_patches = np.reshape(y_patches,(8, 128,128,-1))

            
            y_patch = copy.deepcopy(y_patches)
            x_patches = create(0.85, y_patch, 8)
            
            for k in range(len(x_patches)):
                x[cnt] = x_patches[k]
                y[cnt] = y_patches[k]
                cnt += 1

def curate_(path, lt,  x, y):

    cnt = 0
    for idx, j in tqdm(enumerate(lt)):
        onlyfiles   = [f for f in listdir(lt[idx]) if isfile(join(lt[idx], f))]

        for _, i in enumerate(onlyfiles):
            p = join(str(lt[idx]),i)
            img = cv2.imread(p, 0)
            y_patches = image.extract_patches_2d(img, (128,128), max_patches = 2)

            y_patches = np.reshape(y_patches,(2, 128,128,-1))


            y_patch = copy.deepcopy(y_patches)
            x_patches = create(0.85, y_patch, 2)

            for k in range(len(x_patches)):
                x[cnt] = x_patches[k]
                y[cnt] = y_patches[k]
                cnt += 1

def imgSave():
    
    x_ = np.zeros((10, 128, 128, 1), dtype=np.uint8)
    y_ = np.zeros((10, 128, 128, 1), dtype=np.uint8)

    data = Path('/workspace/storage/cnn-cs/data/test')
    lst = os.listdir(data)
    lst.sort()

    count = 0

    for _, i in enumerate(lst):
        p = join(data,i)
        img = cv2.imread(p, 0)

        y_patches = image.extract_patches_2d(img, (128, 128), max_patches = 1)
        y_patches = np.reshape(y_patches,(1, 128, 128,-1))
        y_patch = copy.deepcopy(y_patches)
        x_patches = create(0.10, y_patch, 1)

        for k in range(len(x_patches)):
            x_[count] = x_patches[k]
            y_[count] = y_patches[k]
            count += 1

    x_ = x_ / 255.
    y_ = y_ / 255.

    fig = plt.figure(figsize=(25, 25))
    columns = 10
    rows = 1
    for i in range(1, columns*rows + 1):
        img_x = x_[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(np.reshape(img_x,(128,128)),cmap='gray')

    plt.savefig('/workspace/data/image_sparse.png')
    
    model = tf.keras.models.load_model('/workspace/data/cs-simple-model-1000.h5', compile =False)
    
    predict = model.predict(x_[:10,:,:,:])
    
    fig = plt.figure(figsize=(25, 25))
    columns = 10
    rows = 1
    for i in range(1, columns*rows + 1):
        img_x = predict[i-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(np.reshape(img_x,(128,128)), cmap='gray')

    plt.savefig('/workspace/data/image_recons.png')
    psnr_ = tf.image.psnr(y_[:10,:,:,:], predict, max_val=1.0)
    print(psnr_)

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

