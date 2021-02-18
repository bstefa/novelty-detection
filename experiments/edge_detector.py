#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:38:50 2020

@author: brahste
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
# import skimage as ski
import cv2 as cv
import imageio as io

    
from skimage import io, feature, color, segmentation, morphology
# import skimage as ski
import os

from pathlib import Path

# Define constants
DATA_PATH = "/home/brahste/Datasets/LunarAnalogue/images-screened/configurations"

def get_list_of_image_paths(data_path: str, glob_pattern: str="*"):
    return list(Path(data_path).glob(glob_pattern))

def bgr2rgb(image):
    return image[...,::-1]

list_of_image_paths = get_list_of_image_paths(DATA_PATH, "**/all*/**/*.jpeg")
print(f"Found {len(list_of_image_paths)} images...")
im = io.imread(list_of_image_paths[1])

plt.imshow(im)
plt.show()

imcv = cv.imread(str(list_of_image_paths[2]))

im8 = cv.resize(imcv, None, fx=0.2, fy=0.2)

plt.imshow(im8)
plt.show()

#%%

lab8 = cv.cvtColor(im8, cv.COLOR_BGR2LAB)

retval = cv.ximgproc.createSuperpixelSLIC(
    cv.GaussianBlur(lab8, (5,5), 2), 
    cv.ximgproc.MSLIC, 
    region_size=50
)
retval.iterate()

labels = retval.getLabels()
contour_mask = retval.getLabelContourMask()
print(labels.shape)
# print(labels[0], len(labels[0]))
#print(np.unique(labels))
print('cont mask', contour_mask.shape)
print('cont max min', np.max(contour_mask), np.min(contour_mask))
print('cont max min', np.max(lab8), np.min(lab8))
print('number of superpixels: ', retval.getNumberOfSuperpixels())
fig, ax = plt.subplots(1, 1)
#ax.imshow(contour_mask)
ax.imshow(labels)
#ax.imshow(lab8, alpha=0.5)
plt.show()

plt.imshow(contour_mask)
plt.show()

print(retval.getLabels(), type(retval), dir(retval))

#%%

glcmat = feature.greycomatrix(labels, [1], [0, np.pi/2, np.pi, 3*np.pi/2], retval.getNumberOfSuperpixels())
print(glcmat.shape)
tot = np.zeros((retval.getNumberOfSuperpixels(), retval.getNumberOfSuperpixels()))
for i in range(0, 4, 1):
    print(glcmat[:, :, 0, i])
    tot += glcmat[:, :, 0, i]
print(tot/4) # I think this average is acceptable as 'connectivity' of the pixels
#%%

grey8 = cv.cvtColor(im8, cv.COLOR_BGR2GRAY)
equa8 = cv.equalizeHist(grey8)

gauss8 = cv.GaussianBlur(equa8, (9,9), 7)
medi8 = cv.medianBlur(equa8, 11)
thres8 = cv.threshold(medi8, 0, 255, cv.THRESH_OTSU)
#

print(im512.shape)
print(np.max(im8), np.min(im8))

print(type(im8))

plt.imshow(bgr2rgb(imcv)); plt.show()
plt.imshow(bgr2rgb(im8)); plt.show()
plt.imshow(equa8); plt.show()
plt.imshow(gauss8); plt.show()
plt.imshow(medi8); plt.show()

plt.imshow(thres8[1]); plt.show()
# grey8 = cv.cvtColor(medi8, cv.COLOR_BGR2GRAY)
dilate_thres8 = cv.erode(thres8[1], (5,5))
plt.imshow(dilate_thres8); plt.show()
#%%

canny8 = cv.Canny(gauss8, 50, 150)
dial8= cv.dilate(canny8, (15,15))
#flood8 = cv.floodFill(dial8, None, (5, 5), 200)
flood8_2 = cv.floodFill(dial8, None, (410, 240), 200)

plt.imshow(canny8); plt.show()
plt.imshow(dial8); plt.show()
#plt.imshow(flood8); plt.show()

#%%


print(im.shape)
plt.imshow(im)
plt.show()
grey = color.rgb2gray(im)
plt.imshow(grey)
plt.show()
gauss = ski.filters.gaussian(grey, sigma=8)
plt.imshow(gauss)
plt.show()

# # Initial level set
# init_ls = segmentation.checkerboard_level_set(grey.shape, 6)
# ls = segmentation.morphological_chan_vese(gauss, 20, init_level_set=init_ls, smoothing=3)
# #%%
# plt.contour(ls)
# plt.show()


# flood= morphology.flood(ls, (10,10), connectivity=1)
# plt.imshow(flood)
# plt.show()
#%%

canny = feature.canny(grey, sigma=4)
plt.imshow(canny)
plt.show()
dial = ski.morphology.binary_dilation(canny, selem=morphology.disk(4))
plt.imshow(dial)
plt.show()

flood= morphology.flood(dial, (0,0), connectivity=1)
plt.imshow(flood)
plt.show()

dist = ndi.distance_transform_edt(dial)
plt.imshow(dist)
plt.show()
#return 0;

# if __name__ == "__main__":
#     main()