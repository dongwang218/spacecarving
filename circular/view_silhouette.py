import numpy as np
import cv2
import sys
import os

img_file = sys.argv[1]
mask_file = sys.argv[2]

img = cv2.imread(img_file)
mask = np.load(mask_file).astype(np.uint8)*255


cv2.imshow('original', img)
cv2.imshow('mask', mask)
cv2.waitKey()
