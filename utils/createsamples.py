from sklearn.feature_extraction import image
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def create(p):
    
    lst = os.listdir(p)

    for j in range(len(lst)):
        y_patches = image.extract_patches_2d(cv2.imread(os.path.join(str(p),lst[j]),0), (128,128), max_patches = 10)

        for i in range(len(y_patches)):
            plt.imsave(lst[j][:-4]+'_'+str(i)+'.png', y_patches[i],cmap='gray')
    
    return None

if __name__ == "__main__":

    path = Path('/home/wilfred/Datasets/test_images')
    create(path)
