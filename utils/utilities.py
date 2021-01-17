import os
import cv2
import numpy as np
from os import listdir
from pathlib import Path
from random import randint, sample
from os.path import isfile, join

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#define the path for the dataset
data        = Path('/home/wilfred/Datasets/BSR_bsds500/BSR/BSDS500/data/images')
lst         = [x for x in data.iterdir() if data.is_dir()]
cnt = 0
for idx, j in enumerate(lst):
    onlyfiles   = [f for f in listdir(lst[idx]) if isfile(join(lst[idx], f))]
    onlyfiles.remove('Thumbs.db')
    cnt += len(onlyfiles)
    for _, i in enumerate(onlyfiles):
        p = join(str(lst[idx]),i)
        print(p)
        img = mpimg.imread(p)
        print(img.shape)
        #imgplot = plt.imshow(img)
        #plt.show()
    print()

#find the pixels that need to be removed
mask = np.random.choice([0, 1], size=(512, 512), p=[.5, .5])
print(mask)
print(np.where(mask==1))
