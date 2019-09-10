# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:58:14 2019

@author: YesAB
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path
import os
# =============================================================================
# import os
# script_path = os.path.dirname(os.path.abspath( __file__ ))
# parent_dir= os.path.abspath(os.path.join(script_path, os.pardir))
# =============================================================================

# print "file is --->:" + os.path.dirname(os.getcwd())
#parent_dir=os.path.dirname(os.getcwd())

# =============================================================================
# def create_detector(DETECTOR, image):
#     keypoints, descriptors = detector.detectAndCompute(image, None)
#     return keypoints, descriptors
# 
# =============================================================================
wdir=os.getcwd()
img1 = cv2.imread(wdir +"\img\src_02.png",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(wdir +"\img\\tar_02.png",cv2.IMREAD_GRAYSCALE) # Is this a trick (tar??)


# Rescaling Images:
scale_percent = 40 # percent of original size

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image 1 # resize image 2
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


# ORB Detector
orb = cv2.ORB_create(nfeatures=3000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


# SIFT Detector
# =============================================================================
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# =============================================================================

# =============================================================================
# for d in des1:
#     print(d)
# =============================================================================

# Brute Force Matching:
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

print("Number of matches: " + str(len(matches))) #177

matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:], None, flags = 2)

# =============================================================================
# for m in matches:
#     print(m.distance)
# =============================================================================


#cv2.imshow("Image_1",img1)
#cv2.imshow("Image_2",img2)
print "Matching_result: "
plt.figure(figsize=(15,15))
plt.imshow(matching_result)
plt.show()

cv2.imshow(wdir +"\img\Matching_result", matching_result)                

cv2.waitKey(3500) # Wait for 3.5s and destroy
cv2.destroyAllWindows()