#!/usr/bin/env python
'''
refine mask by grabcut
'''

import numpy as np
import cv2
import sys
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--mask_in_dir', default='../dong/masks')
ap.add_argument('--mask_out_dir', default='../dong/masks2')
ap.add_argument('--mask_img_dir', default='../dong/masks2_view')
ap.add_argument('images', nargs='*')
args = ap.parse_args()


for img_file in args.images:
  basename = os.path.basename(img_file).split('.')[0]
  mask_file = os.path.join(args.mask_in_dir, basename+'.npy')
  mask_out_file = os.path.join(args.mask_out_dir, basename+'.npy')
  mask_img_file = os.path.join(args.mask_img_dir, basename+'.jpg')

  img = cv2.imread(img_file)
  mask = np.load(mask_file).astype(np.uint8)*3

  rect = (0,0,1,1)
  bgdmodel = np.zeros((1,65),np.float64)
  fgdmodel = np.zeros((1,65),np.float64)
  cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
  mask2 = np.logical_or((mask==1), (mask==3))
  np.save(mask_out_file, mask2)
  cv2.imwrite(mask_img_file, mask2.astype(np.uint8)*255)
