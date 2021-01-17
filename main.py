import numpy as np
import os
import cv2
from os import listdir
from pathlib import Path
from random import randint, sample
from os.path import isfile, join

data = Path("../../../../Datasets/BSR_bsds500/BSR/BSDS500/data/images")
lst = [x for x in data.iterdir() if data.is_dir()]
print(len(lst))
onlyfiles = [f for f in listdir(lst[0]) if isfile(join(lst[0], f))]

#height = sample(range(0,1024), 512)
#width = sample(range(0,1024), 512)
print(lst[0])
print(len(onlyfiles))
print(sorted(onlyfiles))
#print(sorted(height))
#print(sorted(width))
