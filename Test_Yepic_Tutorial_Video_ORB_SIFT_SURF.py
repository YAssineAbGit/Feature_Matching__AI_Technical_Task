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
orb_kp1, orb_des1 = orb.detectAndCompute(img1, None)
orb_kp2, orb_des2 = orb.detectAndCompute(img2, None)

# SIFT Detector
sift = cv2.xfeatures2d.SIFT_create()
sift_kp1, sift_des1 = sift.detectAndCompute(img1, None)
sift_kp2, sift_des2 = sift.detectAndCompute(img2, None)

# SURF Detector
surf = cv2.xfeatures2d.SURF_create()
surf_kp1, surf_des1 = surf.detectAndCompute(img1, None)
surf_kp2, surf_des2 = surf.detectAndCompute(img2, None)

# =============================================================================
# for d in des1:
#     print(d)
# =============================================================================

# Brute Force Matching:
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher()

orb_matches = bf.match(orb_des1,orb_des2)
sift_matches = bf.knnMatch(sift_des1,sift_des2,k=2)
surf_matches = bf.knnMatch(surf_des1,surf_des2,k=2)


orb_matches = sorted(orb_matches, key = lambda x:x.distance)

# Apply ratio test
sift_good = []
for m,n in sift_matches:
    if m.distance < 0.75*n.distance:
        sift_good.append([m])

# Apply ratio test        
surf_good = []
for m,n in surf_matches:
    if m.distance < 0.75*n.distance:
        surf_good.append([m])


ORB_matching_result = cv2.drawMatches(img1, orb_kp1, img2, orb_kp2, orb_matches[:], None, flags = 2)        
# cv.drawMatchesKnn expects list of lists as matches.
SIFT_matching_result = cv2.drawMatchesKnn(img1,sift_kp1,img2,sift_kp2,sift_good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
SURF_matching_result = cv2.drawMatchesKnn(img1,surf_kp1,img2,surf_kp2,surf_good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# =============================================================================
# for m in matches:
#     print(m.distance)
# =============================================================================


#cv2.imshow("Image_1",img1)
#cv2.imshow("Image_2",img2)
print("Number of orb_matches: " + str(len(orb_matches)))
print "ORB_Matching_result: "
plt.figure(figsize=(15,15))
plt.imshow(ORB_matching_result)
plt.show()

print("Number of sift_good: " + str(len(sift_good)))
print "SIFT_Matching_result: "
plt.figure(figsize=(15,15))
plt.imshow(SIFT_matching_result)
plt.show()

print("Number of surf_good: " + str(len(surf_good)))
print "SURF_Matching_result: "
plt.figure(figsize=(15,15))
plt.imshow(SURF_matching_result)
plt.show()

# =============================================================================
# cv2.imshow(wdir +"\img\Matching_result", matching_result)                
# 
# cv2.waitKey(3500) # Wait for 3.5s and destroy
# cv2.destroyAllWindows()
# =============================================================================
