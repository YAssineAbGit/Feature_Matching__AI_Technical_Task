# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:58:14 2019

@author: YesAB
"""

import cv2
import numpy as np
import os.path
import os
# =============================================================================
# import os
# script_path = os.path.dirname(os.path.abspath( __file__ ))
# parent_dir= os.path.abspath(os.path.join(script_path, os.pardir))
# =============================================================================

print "file is --->:" + os.path.dirname(os.getcwd())
#parent_dir=os.path.dirname(os.getcwd())
wdir=os.getcwd()
img1 = cv2.imread(wdir +"\img\src_02.png",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(wdir +"\img\\tar_02.png",cv2.IMREAD_GRAYSCALE)


# Rescaling Images:
scale_percent = 40 # percent of original size

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image 1
img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

# resize image 2
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# =============================================================================
# for d in des1:
#     print(d)
# =============================================================================

# Brute Force Matching:
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

#print(len(matches)) 177

matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags = 2)

# =============================================================================
# for m in matches:
#     print(m.distance)
# 
# =============================================================================

# the images are too big, we need to resize them: 
# Search on internet how to resize images in OpenCV
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

#cv2.imshow("Image_1",img1)
#cv2.imshow("Image_2",img2)

cv2.imshow("C:\Users\YESSINE AB\Yepic Project\img\Matching_result", matching_result)                

cv2.waitKey(0)
cv2.destroyAllWindows()
